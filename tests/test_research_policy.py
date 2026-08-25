import unittest

from research_policy import (
    QualityScores,
    QualityThresholds,
    extract_tagged_answer,
    quality_passes,
    run_feedback_loop,
)


class QualityPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.thresholds = QualityThresholds()

    def test_all_metrics_are_required(self) -> None:
        self.assertFalse(
            quality_passes(
                QualityScores(answer_relevance=0.9, context_relevance=0.9),
                self.thresholds,
            )
        )

    def test_boundary_values_pass(self) -> None:
        self.assertTrue(
            quality_passes(
                QualityScores(0.60, 0.50, 0.60),
                self.thresholds,
            )
        )

    def test_tag_parser_rejects_malformed_output(self) -> None:
        self.assertEqual(extract_tagged_answer("answer only"), "")
        self.assertEqual(extract_tagged_answer("<answer> revised query </answer>"), "revised query")

    def test_refined_query_is_actually_executed(self) -> None:
        calls: list[str] = []

        def run_once(query: str):
            calls.append(query)
            if query == "original":
                return "weak", QualityScores(0.2, 0.2, 0.2), None
            return "strong", QualityScores(0.9, 0.9, 0.9), None

        outcome = run_feedback_loop(
            "original",
            run_once,
            lambda query, scores: "refined",
            self.thresholds,
            max_refinements=2,
        )
        self.assertEqual(calls, ["original", "refined"])
        self.assertTrue(outcome.passed)
        self.assertEqual(outcome.response, "strong")

    def test_loop_stops_when_refinement_stalls(self) -> None:
        calls: list[str] = []

        def run_once(query: str):
            calls.append(query)
            return "weak", QualityScores(0.2, 0.2, 0.2), None

        outcome = run_feedback_loop(
            "same",
            run_once,
            lambda query, scores: query,
            self.thresholds,
            max_refinements=3,
        )
        self.assertEqual(calls, ["same"])
        self.assertFalse(outcome.passed)


if __name__ == "__main__":
    unittest.main()
