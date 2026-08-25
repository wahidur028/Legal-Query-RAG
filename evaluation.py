"""Direct TruLens RAG-triad scoring without framework instrumentation."""

from __future__ import annotations

import math
from typing import Any

from research_policy import QualityScores


def _node_text(scored_node: Any) -> str:
    node = getattr(scored_node, "node", scored_node)
    get_content = getattr(node, "get_content", None)
    if callable(get_content):
        text = get_content()
    else:
        text = getattr(node, "text", "")
    return str(text or "").strip()


def retrieved_contexts(response: Any) -> list[str]:
    """Return the exact non-empty contexts attached to a query response."""

    return [
        text
        for scored_node in (getattr(response, "source_nodes", None) or [])
        if (text := _node_text(scored_node))
    ]


def _numeric_score(result: Any, metric: str) -> float:
    """Extract and validate the numeric part of a TruLens direct-call result."""

    value = result[0] if isinstance(result, (tuple, list)) else result
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{metric} returned a non-numeric score: {value!r}") from exc
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{metric} returned an invalid score: {score!r}")
    return score


def evaluate_response(provider: Any, query: str, response: Any) -> QualityScores:
    """Evaluate one existing response with the three TruLens RAG-triad metrics.

    Direct provider calls avoid the incompatible TruLens-LlamaIndex recorder package.
    Context relevance is scored per retrieved chunk and averaged, matching the
    previous recorder's aggregation rule.
    """

    answer = str(response).strip()
    contexts = retrieved_contexts(response)
    if not answer:
        raise ValueError("The query engine returned an empty answer.")
    if not contexts:
        raise ValueError("The query engine returned no retrieved context to evaluate.")

    answer_relevance = _numeric_score(
        provider.relevance_with_cot_reasons(query, answer),
        "answer relevance",
    )
    groundedness = _numeric_score(
        provider.groundedness_measure_with_cot_reasons(
            "\n\n".join(contexts),
            answer,
        ),
        "groundedness",
    )
    context_scores = [
        _numeric_score(
            provider.context_relevance_with_cot_reasons(query, context),
            "context relevance",
        )
        for context in contexts
    ]
    context_relevance = math.fsum(context_scores) / len(context_scores)

    return QualityScores(
        answer_relevance=answer_relevance,
        context_relevance=context_relevance,
        groundedness=groundedness,
    )
