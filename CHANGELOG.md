# Changelog

## Unreleased repaired release

- Removed a publicly embedded notebook credential from the current tree.
- Moved the runnable application and dependency file to the repository root.
- Replaced the retired Groq model identifier with a current production model.
- Made the retrieve-evaluate-refine loop execute refined queries and added deterministic regression tests.
- Required all prespecified quality metrics instead of treating partial metrics as sufficient.
- Added source labels, abstention behavior, local-only serving defaults, file-size limits, and legal/confidentiality warnings.
- Stopped deleting the bundled `data/` directory and stopped moving Gradio temporary uploads.
- Derived the FAISS dimension from the selected embedding model instead of hard-coding it.
- Removed unused and source-installed dependencies; pinned a dated compatibility set.
- Removed the unnecessary CrewAI wrapper around the single Groq refinement prompt, preserving the same model role and bounded output contract without its incompatible OpenAI 2.x dependency.
- Replaced the incompatible TruLens-LlamaIndex recorder with supported direct TruLens provider calls over the query response and its retrieved source nodes.
- Pinned Gradio, Hugging Face Hub, Transformers, LlamaIndex OpenAI-like, and OpenAI SDK versions as one resolver-tested compatibility set.
- Included the HTTPX SOCKS transport dependency so the application imports correctly in proxied environments.
- Stopped immediately when evaluation is unavailable instead of spending additional generation calls on refinement without evaluator feedback.
- Added deterministic tests for direct RAG-triad score extraction, context aggregation, invalid scores, missing context, and evaluator-failure behavior.
- Added CI, Dependabot, security guidance, contribution guidance, citation metadata, and repository auditing.
- Documented duplicate PDFs and remaining reproducibility limitations.
