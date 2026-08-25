# Reproducibility status

## What this repository currently supports

- A runnable demonstration of hybrid sparse/dense retrieval, recursive chunks, reranking, response synthesis, LLM-based evaluation, and query refinement.
- Resolver-tested direct and compatibility pins dated 2026-08-25.
- Deterministic tests for threshold and feedback-loop control logic.
- Static repository checks for common secrets, invalid notebooks, and duplicate PDFs.

## What it does not yet establish

The repository does not currently contain the complete benchmark datasets, immutable split manifests, fine-tuned model checkpoints with hashes, training configurations, per-run seeds, raw predictions, hardware logs, metric scripts, statistical analysis, or a command that reconstructs every table and figure in the IEEE Access paper.

The percentages shown in the README are therefore reported-paper results, not results independently reproduced by the present repair.

## Minimum publication-grade reproduction package

Add the following without using the protected test set for development:

1. Dataset provenance, licenses, checksums, and immutable train/development/test manifests.
2. Exact embedding, generator, evaluator, and reranker checkpoints and revisions.
3. Training and inference configurations, random seeds, library lockfile, hardware, and environment capture.
4. Baseline implementations with equal retrieval corpus, tuning budget, and evaluation opportunity.
5. Raw ranked lists, generated answers, evaluator outputs, latency/cost logs, and failure logs.
6. Metric definitions and scripts for Hit Rate, MRR, answer relevance, context relevance, and groundedness.
7. Prespecified uncertainty analysis, multiple-seed or bootstrap plan, and correction for repeated comparisons when applicable.
8. A single command that regenerates result tables and figures from immutable raw artifacts.

## Interpretation rule

LLM-judge scores are measurements from a fallible evaluator, not ground truth. Record the evaluator model, prompt, temperature, provider version, and repeated-judge reliability. Validate judge decisions against human annotations on a protected sample before using thresholds as scientific evidence.

The runtime makes direct TruLens provider calls and does not create a TruLens trace database. Set `OPENAI_EVALUATOR_MODEL` explicitly and retain application logs and raw experiment outputs for any formal evaluation run.
