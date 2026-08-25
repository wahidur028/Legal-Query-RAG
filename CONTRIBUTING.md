# Contributing

## Before opening a pull request

1. Create a branch from `main`.
2. Never commit `.env`, API keys, user uploads, model caches, or legal client data.
3. Keep research claims separate from implementation observations.
4. Add or update a deterministic test for changes to decision logic.
5. Run:

```bash
python -m compileall -q app.py evaluation.py prompt.py query_agents.py research_policy.py scripts tests
python -m unittest discover -s tests -v
python scripts/audit_repository.py
```

## Research changes

For changes that claim an improvement, document the dataset and split, baselines, model versions, prompts, seeds, hardware, metric definitions, uncertainty analysis, and whether the evaluation set influenced development. Do not replace reported paper results with new numbers unless the full evidence is available.

## Pull-request description

Explain the problem, the change, the validation performed, remaining limitations, and whether the change affects compatibility with the published paper.
