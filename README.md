# Legal Query RAG (LQ-RAG)

[![Validate repository](https://github.com/wahidur028/Legal-Query-RAG/actions/workflows/validate.yml/badge.svg)](https://github.com/wahidur028/Legal-Query-RAG/actions/workflows/validate.yml)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FACCESS.2025.3542125-blue)](https://doi.org/10.1109/ACCESS.2025.3542125)

Official research-code repository for **“Legal Query RAG”**, published in *IEEE Access* 13 (2025), 36978–36994.

LQ-RAG combines dense and BM25 retrieval, recursive chunks, cross-encoder reranking, response synthesis, LLM-based quality evaluation, and evaluator-guided query refinement.

> **Research and safety notice:** This is a research prototype. It can be wrong, is not legal advice, and must not be used as the sole basis for legal decisions. Do not upload confidential or privileged material to an untrusted deployment.

## What was repaired

The original public tree had a broken installation path, a retired Groq model ID, an embedded notebook credential, no CI, and a feedback loop that refined a failed query but exited before executing the refined query. The repaired version provides a root-level runnable app, current pinned dependencies, secret-safe configuration, deterministic loop tests, CI, repository auditing, source labels, conservative abstention, and local-only serving by default.

See [CHANGELOG.md](CHANGELOG.md) for the complete change list and [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the evidence boundary.

## System flow

1. Copy uploaded PDFs into ignored runtime storage.
2. Split each document into base and recursive sub-chunks.
3. Retrieve with dense FAISS and sparse BM25 retrievers.
4. Deduplicate and rerank retrieved nodes.
5. Generate an answer only from retrieved context.
6. Evaluate answer relevance, context relevance, and groundedness.
7. If all prespecified thresholds are not met, pass the failed scores to the query-refinement agent and execute the refined query.
8. Stop on pass, stalled refinement, or the refinement budget; otherwise abstain by default.

The optional model-only fallback is disabled because low retrieval support should not silently become unsupported legal guidance.

## Requirements

- Python 3.11
- A Groq API key for generation and query refinement
- An OpenAI API key for the TruLens evaluator
- Internet access on first run to download the embedding and reranker models
- Enough memory for the selected embedding and reranker models

## Quick start

```bash
git clone https://github.com/wahidur028/Legal-Query-RAG.git
cd Legal-Query-RAG

python -m venv .venv
source .venv/bin/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

cp .env.example .env       # Windows: copy .env.example .env
# Edit .env and add GROQ_API_KEY and OPENAI_API_KEY.

python app.py
```

Open `http://127.0.0.1:7860`, upload one or more PDFs, click **Upload and index**, and then ask a question.

The default configuration does not create a public Gradio share link. Set `GRADIO_SHARE=true` only when you understand that uploaded documents and model outputs may be exposed through that deployment.

## Configuration

Copy [.env.example](.env.example) to `.env`. Never commit `.env`.

| Variable | Default | Purpose |
|---|---:|---|
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Current Groq production generator/refiner model |
| `EMBEDDING_MODEL` | `avsolatorio/GIST-large-Embedding-v0` | Dense embedding model |
| `RERANKER_MODEL` | `BAAI/bge-reranker-large` | Cross-encoder reranker |
| `MAX_REFINEMENTS` | `2` | Maximum refined queries after the original query |
| `MIN_ANSWER_RELEVANCE` | `0.60` | Required evaluator score |
| `MIN_CONTEXT_RELEVANCE` | `0.50` | Required evaluator score |
| `MIN_GROUNDEDNESS` | `0.60` | Required evaluator score |
| `ALLOW_UNGROUNDED_FALLBACK` | `false` | Permit model-only fallback; unsafe for most legal uses |
| `GRADIO_SHARE` | `false` | Create a public share link |

The thresholds reproduce the original prototype's nominal cutoffs, not universally validated legal-safety guarantees. Calibrate them on a protected, human-annotated dataset before making performance or safety claims.

## Reported paper results

The associated paper reports improvements in Hit Rate, MRR, model-task performance, and relevance relative to its stated baselines. Those are **reported-paper results**. This repository does not yet contain the complete immutable datasets, run manifests, raw predictions, seeds, statistical analysis, and one-command benchmark pipeline required for an independent reproduction of every number.

Do not present a successful app launch or a green CI check as reproduction of the paper. CI currently validates repository structure, secret hygiene, Python syntax, and deterministic decision logic.

## Repository layout

```text
.
├── app.py                     # Gradio application and RAG orchestration
├── crew_ai.py                 # Lazy query-refinement and optional fallback agents
├── prompt.py                  # Document-grounded synthesis prompt
├── research_policy.py         # Pure thresholds and feedback-loop control logic
├── requirements.txt           # Dated dependency snapshot
├── notebooks/                 # Original demonstration and fine-tuning notebooks
├── data/                      # Bundled PDFs plus dataset card
├── scripts/audit_repository.py
├── tests/test_research_policy.py
└── .github/workflows/validate.yml
```

## Validate locally

These checks do not require API keys or model downloads:

```bash
python -m compileall -q app.py crew_ai.py prompt.py research_policy.py scripts tests
python -m unittest discover -s tests -v
python scripts/audit_repository.py
```

The audit reports exact duplicate PDFs as warnings. See [data/DATASET_CARD.md](data/DATASET_CARD.md).

## Credential warning for existing clones and forks

An earlier notebook revision contained a Hugging Face token. The repaired tree removes it, but Git history and forks can retain old content. The token owner must revoke it; deleting the visible line is not sufficient. See [SECURITY.md](SECURITY.md).

## Citation

```bibtex
@ARTICLE{10887211,
  author={Wahidur, Rahman S. M. and Kim, Sumin and Choi, Haeung and Bhatti, David S. and Lee, Heung-No},
  journal={IEEE Access},
  title={Legal Query RAG},
  year={2025},
  volume={13},
  pages={36978--36994},
  doi={10.1109/ACCESS.2025.3542125}
}
```

GitHub-compatible citation metadata is also provided in [CITATION.cff](CITATION.cff).

## Licensing status

The previous README displayed a CC BY 4.0 badge, but the repository did not contain a license file. A public repository without an explicit license is not automatically open source. The copyright holders should agree on and commit the license before the repaired release is described as open source. A common structure is a software license such as Apache-2.0 or MIT for code and a separate content/data license only where redistribution rights are established.
