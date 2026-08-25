# Repair audit

Audit date: 2026-08-25

## Directly observed defects in the previous public tree

- Root README commands referenced root `requirements.txt` and `app.py`, but both files were nested under `legal-query-rag/`.
- A notebook contained a Hugging Face access token in executable source.
- The configured Groq generator model was no longer listed as a current production model.
- The code hard-coded a 1024-dimensional FAISS index instead of deriving the dimension from the selected embedding model.
- PDF handling deleted the active data directory and moved Gradio temporary uploads, which could fail across filesystems and destroy bundled data.
- The query was executed twice before evaluation.
- `max_attempts` was one; the code refined a failed query and then exited without executing the refined query.
- Missing evaluator metrics could still allow a response to pass through partial-metric branches.
- Low evidence triggered a model-only legal answer by default.
- Gradio launched through a hard-coded public share server.
- Dependencies were unpinned, included unused packages, and installed `bitsandbytes` directly from a Git repository.
- The data directory contained four exact duplicate-PDF groups and no provenance record.
- No GitHub Actions workflow, security policy, release, or license file was present.

## Repairs implemented

- Root-level runnable layout and corrected quick start.
- Secret removed from the current notebook tree; `.env.example` and `.gitignore` added.
- Current Groq production model identifier and dated dependency snapshot.
- Dynamic FAISS dimension, runtime-only upload directory, safe copying, and local-only serving default.
- One query execution per evaluated attempt; refined queries are executed within a bounded loop.
- Strict all-metric pass rule; missing evaluation is reported as unverified.
- Evidence-first abstention by default; model-only fallback requires explicit opt-in.
- Retrieved source labels included with responses where metadata is available.
- Cleared notebook outputs and replaced machine-specific paths and key placeholders.
- Deterministic tests, repository audit, CI, Dependabot, security policy, contribution guide, dataset card, reproducibility boundary, changelog, and citation metadata.

## Validation completed

- Python compilation passed for application, policy, scripts, and tests.
- Five deterministic unit tests passed.
- Repository safety/structure audit passed.
- Secret-pattern scan found no remaining Hugging Face, OpenAI-style, or Groq tokens in the repaired tree.
- Notebook JSON schemas parsed successfully.
- Four duplicate-PDF groups remain intentionally reported as warnings and documented in the dataset card.

## Validation not claimed

No end-to-end API/model integration run was performed because it requires user-owned Groq and OpenAI credentials, large model downloads, and external inference. The GitHub workflow performs dependency resolution in Python 3.11; after upload, that step must be green before release. Passing CI does not reproduce the paper's numerical results.

## Human actions still required

1. Revoke the exposed Hugging Face token; current-tree deletion is insufficient.
2. Decide and commit a license with authorization from the relevant copyright holders.
3. Upload the repaired tree and confirm the GitHub Actions workflow is green.
4. Run a protected end-to-end smoke test with non-confidential PDFs and test-only API keys.
