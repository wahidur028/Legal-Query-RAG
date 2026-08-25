import unittest

from evaluation import evaluate_response, retrieved_contexts


class FakeNode:
    def __init__(self, text: str) -> None:
        self.text = text

    def get_content(self) -> str:
        return self.text


class FakeScoredNode:
    def __init__(self, text: str) -> None:
        self.node = FakeNode(text)


class FakeResponse:
    def __init__(self, answer: str, contexts: list[str]) -> None:
        self.answer = answer
        self.source_nodes = [FakeScoredNode(text) for text in contexts]

    def __str__(self) -> str:
        return self.answer


class FakeProvider:
    def __init__(self) -> None:
        self.context_calls: list[tuple[str, str]] = []

    def relevance_with_cot_reasons(self, prompt: str, response: str):
        return 0.7, {"reason": "test"}

    def groundedness_measure_with_cot_reasons(self, source: str, statement: str):
        return 0.8, {"reason": "test"}

    def context_relevance_with_cot_reasons(self, question: str, context: str):
        self.context_calls.append((question, context))
        return (0.3 if context == "first context" else 0.9), {"reason": "test"}


class DirectEvaluationTests(unittest.TestCase):
    def test_contexts_are_taken_from_response_source_nodes(self) -> None:
        response = FakeResponse("answer", ["first context", "second context"])
        self.assertEqual(
            retrieved_contexts(response),
            ["first context", "second context"],
        )

    def test_rag_triad_scores_and_averages_each_context(self) -> None:
        provider = FakeProvider()
        response = FakeResponse("answer", ["first context", "second context"])

        scores = evaluate_response(provider, "question", response)

        self.assertEqual(scores.answer_relevance, 0.7)
        self.assertEqual(scores.groundedness, 0.8)
        self.assertAlmostEqual(scores.context_relevance or 0.0, 0.6)
        self.assertEqual(
            provider.context_calls,
            [
                ("question", "first context"),
                ("question", "second context"),
            ],
        )

    def test_missing_context_is_an_evaluation_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "no retrieved context"):
            evaluate_response(FakeProvider(), "question", FakeResponse("answer", []))

    def test_out_of_range_score_is_rejected(self) -> None:
        provider = FakeProvider()
        provider.relevance_with_cot_reasons = lambda prompt, response: (1.5, {})
        with self.assertRaisesRegex(ValueError, "invalid score"):
            evaluate_response(
                provider,
                "question",
                FakeResponse("answer", ["first context"]),
            )


if __name__ == "__main__":
    unittest.main()
