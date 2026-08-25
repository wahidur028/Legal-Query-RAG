"""Gradio application for the Legal Query RAG research prototype.

The application prefers an explicit abstention over an answer that is weakly
supported by the uploaded documents. It is research software, not legal advice.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any

import faiss
import gradio as gr
import numpy as np
from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    get_response_synthesizer,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever, RecursiveRetriever
from llama_index.core.schema import IndexNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.faiss import FaissVectorStore

from crew_ai import generate_ungrounded_answer, refine_query
from prompt import custom_prompt
from research_policy import (
    QualityScores,
    QualityThresholds,
    run_feedback_loop,
)


ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = ROOT / ".runtime"
UPLOAD_DIR = RUNTIME_DIR / "uploads"

load_dotenv(ROOT / ".env")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
LOGGER = logging.getLogger("legal_query_rag")

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "avsolatorio/GIST-large-Embedding-v0"
)
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-large")
MAX_REFINEMENTS = max(0, int(os.getenv("MAX_REFINEMENTS", "2")))
ALLOW_UNGROUNDED_FALLBACK = os.getenv(
    "ALLOW_UNGROUNDED_FALLBACK", "false"
).lower() in {"1", "true", "yes"}

THRESHOLDS = QualityThresholds(
    answer_relevance=float(os.getenv("MIN_ANSWER_RELEVANCE", "0.60")),
    context_relevance=float(os.getenv("MIN_CONTEXT_RELEVANCE", "0.50")),
    groundedness=float(os.getenv("MIN_GROUNDEDNESS", "0.60")),
)

query_engine: RetrieverQueryEngine | None = None
embed_model: HuggingFaceEmbedding | None = None


class HybridRetriever(BaseRetriever):
    """Union BM25 and dense results while removing duplicate nodes."""

    def __init__(self, vector_retriever: Any, bm25_retriever: Any) -> None:
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        super().__init__()

    def _retrieve(self, query_bundle: Any, **kwargs: Any) -> list[Any]:
        sparse_nodes = self.bm25_retriever.retrieve(query_bundle, **kwargs)
        dense_nodes = self.vector_retriever.retrieve(query_bundle, **kwargs)
        unique_nodes: list[Any] = []
        seen: set[str] = set()
        for scored_node in sparse_nodes + dense_nodes:
            node_id = scored_node.node.node_id
            if node_id not in seen:
                unique_nodes.append(scored_node)
                seen.add(node_id)
        return unique_nodes


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing {name}. Copy .env.example to .env and add the required key."
        )
    return value


def _initialize_models() -> tuple[Groq, HuggingFaceEmbedding]:
    global embed_model
    llm = Groq(model=GROQ_MODEL, api_key=_require("GROQ_API_KEY"))
    if embed_model is None:
        embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    Settings.llm = llm
    Settings.embed_model = embed_model
    return llm, embed_model


def _safe_pdf_paths(files: list[Any] | None) -> list[Path]:
    if not files:
        raise ValueError("Upload at least one PDF file.")
    paths = [Path(str(item)).resolve() for item in files]
    invalid = [path.name for path in paths if path.suffix.lower() != ".pdf"]
    missing = [path.name for path in paths if not path.is_file()]
    if invalid:
        raise ValueError(f"Only PDF files are supported: {', '.join(invalid)}")
    if missing:
        raise ValueError(f"Uploaded file is unavailable: {', '.join(missing)}")
    return paths


def process_pdf(files: list[Any] | None) -> str:
    """Copy uploaded PDFs into runtime storage and build the hybrid index."""

    global query_engine
    try:
        paths = _safe_pdf_paths(files)
        _, embedding = _initialize_models()

        if UPLOAD_DIR.exists():
            shutil.rmtree(UPLOAD_DIR)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        copied: list[Path] = []
        for source in paths:
            destination = UPLOAD_DIR / source.name
            shutil.copy2(source, destination)
            copied.append(destination)

        documents = SimpleDirectoryReader(
            input_files=[str(path) for path in copied]
        ).load_data(num_workers=min(4, len(copied)))
        base_nodes = SentenceSplitter(
            chunk_size=512, chunk_overlap=64
        ).get_nodes_from_documents(documents)
        if not base_nodes:
            raise ValueError("No extractable text was found in the uploaded PDFs.")

        all_nodes: list[Any] = []
        for base_node in base_nodes:
            for chunk_size in (128, 256):
                sub_nodes = SentenceSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=min(32, chunk_size // 4),
                ).get_nodes_from_documents([base_node])
                all_nodes.extend(
                    IndexNode.from_text_node(node, base_node.node_id)
                    for node in sub_nodes
                )
            all_nodes.append(IndexNode.from_text_node(base_node, base_node.node_id))

        node_dict = {node.node_id: node for node in all_nodes}
        dimension = len(embedding.get_text_embedding("embedding dimension probe"))
        vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatL2(dimension))
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        storage_context.docstore.add_documents(all_nodes)
        index = VectorStoreIndex(all_nodes, storage_context=storage_context)

        top_k = min(5, len(all_nodes))
        dense = index.as_retriever(similarity_top_k=top_k)
        recursive_dense = RecursiveRetriever(
            "vector",
            retriever_dict={"vector": dense},
            node_dict=node_dict,
            verbose=False,
        )
        sparse = BM25Retriever.from_defaults(
            nodes=base_nodes, similarity_top_k=min(5, len(base_nodes))
        )
        hybrid = HybridRetriever(recursive_dense, sparse)
        reranker = SentenceTransformerRerank(
            top_n=min(3, len(base_nodes)), model=RERANKER_MODEL
        )

        query_engine = RetrieverQueryEngine.from_args(
            retriever=hybrid,
            node_postprocessors=[reranker],
            response_synthesizer=get_response_synthesizer(response_mode="compact"),
            streaming=False,
        )
        query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": custom_prompt}
        )
        return f"Indexed {len(copied)} document(s) into {len(base_nodes)} base chunks."
    except Exception as exc:
        LOGGER.exception("Document indexing failed")
        query_engine = None
        return f"Indexing failed: {exc}"


def _feedback_scores(results: dict[Any, Any]) -> QualityScores:
    values: dict[str, float] = {}
    for definition, result in results.items():
        name = str(getattr(definition, "name", ""))
        value = getattr(result, "result", None)
        if value is not None:
            values[name.lower()] = float(value)
    return QualityScores(
        answer_relevance=values.get("answer relevance"),
        context_relevance=values.get("context relevance"),
        groundedness=values.get("groundedness"),
    )


def _query_and_evaluate(query: str) -> tuple[Any, QualityScores, str | None]:
    """Execute one query and synchronously collect its evaluator results."""

    assert query_engine is not None
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        return query_engine.query(query), QualityScores(), "OPENAI_API_KEY is missing"

    try:
        from trulens.apps.llamaindex import TruLlama
        from trulens.core import Feedback
        from trulens.providers.openai import OpenAI

        provider = OpenAI(api_key=openai_key)
        context = TruLlama.select_context(query_engine)
        feedbacks = [
            Feedback(
                provider.groundedness_measure_with_cot_reasons,
                name="Groundedness",
            ).on(context.collect()).on_output(),
            Feedback(
                provider.relevance_with_cot_reasons,
                name="Answer Relevance",
            ).on_input_output(),
            Feedback(
                provider.context_relevance_with_cot_reasons,
                name="Context Relevance",
            ).on_input().on(context).aggregate(np.mean),
        ]
        recorder = TruLlama(
            query_engine,
            app_name="Legal_Query_RAG",
            app_version="repaired-v1",
            feedbacks=feedbacks,
        )
        with recorder as recording:
            response = query_engine.query(query)
        record = recording.records[-1]
        scores = _feedback_scores(
            record.wait_for_feedback_results(feedback_timeout=180)
        )
        return response, scores, None
    except Exception as exc:
        LOGGER.exception("TruLens evaluation failed")
        return query_engine.query(query), QualityScores(), str(exc)


def _source_list(response: Any) -> str:
    sources: list[str] = []
    seen: set[str] = set()
    for scored_node in getattr(response, "source_nodes", []) or []:
        metadata = getattr(scored_node.node, "metadata", {}) or {}
        name = metadata.get("file_name") or metadata.get("filename") or "uploaded PDF"
        page = metadata.get("page_label") or metadata.get("page_number")
        citation = f"{name}, page {page}" if page is not None else str(name)
        if citation not in seen:
            sources.append(citation)
            seen.add(citation)
    return "\n".join(f"- {source}" for source in sources[:5])


def _render(response: Any, scores: QualityScores, label: str) -> str:
    sources = _source_list(response)
    citation_text = f"\n\nSources retrieved:\n{sources}" if sources else ""
    return f"{label}\n\n{response}\n\nEvaluation: {scores.as_text()}{citation_text}"


def query_model(user_query: str) -> str:
    """Run the retrieve-evaluate-refine loop."""

    if query_engine is None:
        return "Upload and index one or more PDFs before asking a question."
    query = (user_query or "").strip()
    if not query:
        return "Enter a legal question."

    outcome = run_feedback_loop(
        initial_query=query,
        run_once=_query_and_evaluate,
        refine=refine_query,
        thresholds=THRESHOLDS,
        max_refinements=MAX_REFINEMENTS,
    )

    if outcome.passed:
        return _render(outcome.response, outcome.scores, "Document-grounded response")

    if outcome.evaluation_error is not None:
        return _render(
            outcome.response,
            outcome.scores,
            "Evaluation unavailable; this answer is unverified and must not be treated as legal advice.",
        )

    if ALLOW_UNGROUNDED_FALLBACK:
        fallback = generate_ungrounded_answer(user_query)
        return (
            "Ungrounded model-only fallback (disabled by default). The uploaded "
            "documents did not support a satisfactory answer. Verify independently "
            f"and consult a qualified legal professional.\n\n{fallback}"
        )

    return _render(
        outcome.response,
        outcome.scores,
        "Insufficient document support. The system is abstaining; verify the sources and consult a qualified legal professional.",
    )


with gr.Blocks(title="Legal Query RAG") as demo:
    gr.Markdown("# Legal Query RAG")
    gr.Markdown(
        "Research prototype accompanying the IEEE Access paper. Upload PDFs, "
        "retrieve evidence, evaluate the answer, and refine the query when needed."
    )
    gr.Markdown(
        "**Important:** This system can be wrong and is not a substitute for "
        "professional legal advice. Do not upload confidential or privileged material."
    )
    with gr.Row():
        pdf_input = gr.Files(file_types=[".pdf"], label="PDF documents")
        upload_button = gr.Button("Upload and index")
    upload_output = gr.Textbox(label="Index status", interactive=False)
    upload_button.click(process_pdf, inputs=pdf_input, outputs=upload_output)
    user_query = gr.Textbox(label="Legal question")
    submit_button = gr.Button("Ask")
    response_output = gr.Markdown()
    submit_button.click(query_model, inputs=user_query, outputs=response_output)
    user_query.submit(query_model, inputs=user_query, outputs=response_output)


if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        share=os.getenv("GRADIO_SHARE", "false").lower() in {"1", "true", "yes"},
        max_file_size=os.getenv("GRADIO_MAX_FILE_SIZE", "20mb"),
    )
