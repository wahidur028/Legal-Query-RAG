# Legal Query RAG (LQ-RAG)

[![Validate repository](https://github.com/wahidur028/Legal-Query-RAG/actions/workflows/validate.yml/badge.svg)](https://github.com/wahidur028/Legal-Query-RAG/actions/workflows/validate.yml)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FACCESS.2025.3542125-blue)](https://doi.org/10.1109/ACCESS.2025.3542125)

Research-code repository for **“Legal Query RAG,”** published in *IEEE Access* 13 (2025), 36978–36994.

LQ-RAG combines dense and BM25 retrieval, recursive chunks, cross-encoder reranking, document-grounded response synthesis, LLM-based quality evaluation, and evaluator-guided query refinement.

> **Research and safety notice:** This is a research prototype, not legal advice. It can be wrong. Do not use it as the sole basis for a legal decision, and do not upload confidential or privileged material to an untrusted deployment.

## Current validation status

The dependency conflict and feedback-loop defects in the original public prototype have been repaired, but the repository is **not yet certified as a complete reproduction of the paper**.

| Validation layer | Current status | Evidence boundary |
|---|---|---|
| Dependency resolution | Passing | GitHub Actions performs a Python 3.11 resolver dry run |
| Clean local installation | Passing on the repaired Windows/Python 3.11 test environment | This is not yet repeated automatically in CI |
| Python compilation | Passing | Production files, scripts, and tests compile |
| Deterministic logic tests | 10 passing | Evaluation parsing, score validation, thresholds, refinement execution, stalls, and evaluator failures |
| Repository safety audit | Passing after excluding ignored local environments | Checks committable files, required files, notebooks, common credential patterns, and duplicate PDFs |
| Offline end-to-end integration test | Not yet implemented | PDF-to-answer behavior is not yet exercised by one deterministic automated test |
| Live Groq/OpenAI smoke test | Not yet certified | Provider access, model downloads, indexing, generation, and evaluation still require a controlled test |
| Paper-result reproduction | Not established | Complete immutable datasets, manifests, checkpoints, raw outputs, seeds, and benchmark scripts are not present |

A green CI badge proves only the checks implemented in the workflow. It does not prove that every external model and API works, and it does not reproduce the numerical results reported in the paper.

## System flow

1. Copy uploaded PDFs into ignored runtime storage.
2. Parse the documents and create base chunks plus smaller recursive chunks.
3. Retrieve candidates using dense FAISS retrieval and sparse BM25 retrieval.
4. Remove duplicate nodes and rerank the combined candidates with a cross-encoder.
5. Generate an answer from the retrieved document context.
6. Evaluate answer relevance, context relevance, and groundedness for that exact response.
7. If any required score is below its threshold, send the failed scores to the query-refinement prompt and execute the refined query.
8. Stop when all thresholds pass, refinement stalls, evaluation fails, or the configured refinement budget is exhausted.
9. Abstain by default when document support remains insufficient.

The optional model-only fallback is disabled because weak document support must not silently become unsupported legal guidance.

## Requirements

- Python 3.11
- A Groq API key for answer generation and query refinement
- An OpenAI API key for the TruLens evaluator
- Internet access on first run to download the embedding and reranker models
- Sufficient disk space and memory for the selected Hugging Face models
- A non-confidential PDF for smoke testing

## Installation

### Windows Command Prompt

```bat
git clone https://github.com/wahidur028/Legal-Query-RAG.git
cd Legal-Query-RAG

py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

The final command must report:

```text
No broken requirements found.
```

### Linux or macOS

```bash
git clone https://github.com/wahidur028/Legal-Query-RAG.git
cd Legal-Query-RAG

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

## Deterministic validation

Run these commands after installation and before adding API keys:

```bash
python -m compileall -q app.py evaluation.py prompt.py query_agents.py research_policy.py scripts tests
python -m unittest discover -s tests -v
python scripts/audit_repository.py
```

Expected results:

- Compilation completes without output.
- The test runner ends with `Ran 10 tests` and `OK`.
- The audit may report four known duplicate-PDF groups as warnings, but must end with `Repository audit passed`.

These checks do not download models, call external APIs, index a PDF, or reproduce the paper’s experiments.

## Configure API keys

Create a local `.env` file from the provided template:

```bat
copy .env.example .env
```

Linux or macOS:

```bash
cp .env.example .env
```

Edit `.env` and enter new or test-only credentials:

```dotenv
GROQ_API_KEY=
OPENAI_API_KEY=
```

Never commit `.env`, paste keys into an issue, or reuse a previously exposed token. `HF_TOKEN` is optional and should remain empty unless a selected Hugging Face model requires authentication.

## Configuration

| Variable | Default | Purpose |
|---|---:|---|
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq generator and query-refinement model |
| `OPENAI_EVALUATOR_MODEL` | `gpt-4o-mini` | TruLens RAG-triad evaluator model |
| `EMBEDDING_MODEL` | `avsolatorio/GIST-large-Embedding-v0` | Dense embedding model |
| `RERANKER_MODEL` | `BAAI/bge-reranker-large` | Cross-encoder reranker |
| `MAX_REFINEMENTS` | `2` | Maximum refined queries after the original query |
| `MIN_ANSWER_RELEVANCE` | `0.60` | Minimum required evaluator score |
| `MIN_CONTEXT_RELEVANCE` | `0.50` | Minimum required evaluator score |
| `MIN_GROUNDEDNESS` | `0.60` | Minimum required evaluator score |
| `ALLOW_UNGROUNDED_FALLBACK` | `false` | Permit a model-only fallback |
| `GRADIO_SERVER_NAME` | `127.0.0.1` | Local bind address |
| `GRADIO_SERVER_PORT` | `7860` | Local port |
| `GRADIO_SHARE` | `false` | Create a public Gradio share link |
| `GRADIO_MAX_FILE_SIZE` | `20mb` | Maximum upload size |

The score thresholds reproduce the original prototype’s nominal cutoffs. They are not universally validated legal-safety guarantees. Calibrate them using protected human annotations before treating them as scientific or operational decision thresholds.

## Controlled smoke test

Use new test-only API keys and a small, non-confidential PDF.

```bat
python app.py
```

Then:

1. Open `http://127.0.0.1:7860`.
2. Upload the test PDF.
3. Click **Upload and index**.
4. Confirm that the interface reports the number of indexed documents and base chunks.
5. Ask a question whose answer is clearly present in the PDF.
6. Confirm that the response includes an evaluation and retrieved source labels.
7. Ask one question that is not supported by the PDF and verify that the application refines, stops, or abstains instead of silently presenting an unsupported answer as grounded.
8. Stop the server with `Ctrl+C`.

The first run can take time because the embedding and reranker models must be downloaded. A successful launch alone is not a successful smoke test; document indexing, retrieval, generation, evaluation, and failure behavior must all be observed.

## Failure behavior

- Missing or invalid PDFs produce an indexing error and clear the active query engine.
- Missing `GROQ_API_KEY` prevents model initialization.
- Missing `OPENAI_API_KEY` labels the generated response as unverified.
- Evaluator failure stops refinement instead of spending more generation calls without feedback.
- Missing metrics never count as a quality pass.
- Refinement stops when the query does not change or the configured budget is exhausted.
- Unsupported answers are withheld by default unless `ALLOW_UNGROUNDED_FALLBACK=true` is explicitly enabled.

## Repository layout

```text
.
├── app.py                         # Gradio UI and RAG orchestration
├── evaluation.py                  # Direct TruLens RAG-triad evaluation
├── query_agents.py                # Groq query refinement and optional fallback
├── prompt.py                      # Document-grounded synthesis prompt
├── research_policy.py             # Pure threshold and feedback-loop logic
├── requirements.txt               # Direct and compatibility dependency pins
├── notebooks/                     # Historical demonstration and fine-tuning notebooks
├── data/                          # Bundled PDFs and dataset documentation
├── scripts/audit_repository.py    # Repository safety and structure audit
├── tests/test_evaluation.py       # Direct-evaluation regression tests
├── tests/test_research_policy.py  # Feedback-policy regression tests
└── .github/workflows/validate.yml # GitHub Actions validation workflow
```

Runtime uploads, local API keys, virtual environments, caches, and model files must remain outside version control.

## Reproducibility boundary

This repository currently supports inspection and testing of the repaired demonstration pipeline. It does **not** yet provide an independent, one-command reconstruction of every table and figure in the paper.

Publication-grade reproduction additionally requires:

- Dataset provenance, licenses, checksums, and immutable data splits
- Exact model checkpoints and Hugging Face revisions
- Training and inference configurations and random seeds
- A complete environment lock and hardware record
- Baseline implementations under matched evaluation conditions
- Raw retrieved lists, answers, evaluator outputs, latency, cost, and failure logs
- Metric scripts and prespecified statistical analysis
- A one-command benchmark runner that rebuilds tables and figures from immutable artifacts

Live Groq and OpenAI outputs are not bit-for-bit reproducible because hosted models and provider infrastructure can change. Record model names, prompts, configuration, dates, and raw responses for every formal experiment. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the detailed evidence boundary.

## Security

An earlier notebook revision contained a Hugging Face token. Removing it from the current tree or rewriting branch history does not revoke the credential or erase copies held by existing clones and forks. The token owner must revoke it at Hugging Face and inspect provider usage.

The repository audit detects several common credential formats, but automated scanning is not proof that no secret exists. See [SECURITY.md](SECURITY.md) for the incident procedure.

## Release policy

Do not publish a formal release until all of the following are complete:

- The repository audit correction is committed and passes locally and in GitHub Actions.
- CI performs a real clean installation and `pip check`, not only dependency resolution.
- A deterministic offline integration test exercises the complete internal pipeline.
- A controlled live smoke test succeeds with temporary keys and a non-confidential PDF.
- Model revisions, environment information, and test evidence are recorded.
- All relevant copyright holders approve and commit an appropriate license.

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

GitHub-compatible citation metadata is provided in [CITATION.cff](CITATION.cff).

## Licensing status

This public repository does not currently contain an authorized `LICENSE` file and therefore must not be described as open source. The relevant copyright holders should agree on the licensing scope for the software, documentation, notebooks, and bundled data before a release is published. Software licenses such as Apache-2.0 or MIT do not automatically grant redistribution rights for third-party documents or datasets.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Changes affecting research behavior must include deterministic tests and a clear statement of their compatibility with the published method.
