"""Pure decision logic for the retrieve-evaluate-refine loop."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class QualityScores:
    answer_relevance: float | None = None
    context_relevance: float | None = None
    groundedness: float | None = None

    def as_text(self) -> str:
        def render(value: float | None) -> str:
            return "unavailable" if value is None else f"{value:.3f}"

        return (
            f"answer relevance={render(self.answer_relevance)}, "
            f"context relevance={render(self.context_relevance)}, "
            f"groundedness={render(self.groundedness)}"
        )


@dataclass(frozen=True)
class QualityThresholds:
    answer_relevance: float = 0.60
    context_relevance: float = 0.50
    groundedness: float = 0.60


@dataclass(frozen=True)
class LoopOutcome:
    response: Any
    scores: QualityScores
    query: str
    attempts: int
    passed: bool
    evaluation_error: str | None = None


def quality_passes(scores: QualityScores, thresholds: QualityThresholds) -> bool:
    """Require every metric; missing evidence never counts as a pass."""

    values = (
        scores.answer_relevance,
        scores.context_relevance,
        scores.groundedness,
    )
    if any(value is None for value in values):
        return False
    return bool(
        scores.answer_relevance >= thresholds.answer_relevance
        and scores.context_relevance >= thresholds.context_relevance
        and scores.groundedness >= thresholds.groundedness
    )


def extract_tagged_answer(text: str) -> str:
    """Extract one answer tag without silently accepting malformed output."""

    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text or "", re.DOTALL)
    return match.group(1).strip() if match else ""


def run_feedback_loop(
    initial_query: str,
    run_once: Callable[[str], tuple[Any, QualityScores, str | None]],
    refine: Callable[[str, QualityScores], str],
    thresholds: QualityThresholds,
    max_refinements: int,
) -> LoopOutcome:
    """Execute every refined query and stop only on pass, stall, or budget."""

    query = initial_query
    response: Any = None
    scores = QualityScores()
    evaluation_error: str | None = None
    attempts = 0

    for attempt in range(max(0, max_refinements) + 1):
        attempts = attempt + 1
        response, scores, evaluation_error = run_once(query)
        if evaluation_error is None and quality_passes(scores, thresholds):
            return LoopOutcome(response, scores, query, attempts, True, None)
        if attempt >= max_refinements:
            break
        refined = refine(query, scores)
        if not refined or refined == query:
            break
        query = refined

    return LoopOutcome(
        response,
        scores,
        query,
        attempts,
        False,
        evaluation_error,
    )
