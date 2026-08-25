"""Prompt used by the LlamaIndex response synthesizer."""

from llama_index.core import PromptTemplate

custom_prompt = PromptTemplate(
    """You answer only from the supplied legal-document context.

Rules:
1. If the context does not support the answer, say that the documents are insufficient.
2. Do not invent statutes, cases, quotations, dates, jurisdictions, or citations.
3. Separate what the documents state from any uncertainty or limitation.
4. Keep the response concise and professional.
5. This is research output, not legal advice.

Context:
{context_str}

Question:
{query_str}

Document-grounded answer:
"""
)
