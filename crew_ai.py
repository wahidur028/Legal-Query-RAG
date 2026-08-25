"""Lazy CrewAI agents used by the feedback loop."""

from __future__ import annotations

import os

from research_policy import QualityScores, extract_tagged_answer


def _run_agent(
    role: str,
    goal: str,
    description: str,
    topic: str,
    feedback: str,
) -> str:
    from crewai import Agent, Crew, LLM, Task

    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    llm = LLM(
        model=f"groq/{model}",
        temperature=0.2,
        max_tokens=512,
        seed=42,
    )
    agent = Agent(
        role=role,
        goal=goal,
        backstory="A careful research assistant that follows evidence and output constraints.",
        allow_delegation=False,
        llm=llm,
        max_retry_limit=1,
        verbose=False,
    )
    task = Task(
        description=description,
        expected_output="Exactly one result enclosed in <answer>...</answer> tags.",
        agent=agent,
    )
    result = Crew(agents=[agent], tasks=[task], verbose=False).kickoff(
        inputs={"topic": topic, "feedback": feedback}
    )
    outputs = getattr(result, "tasks_output", None) or []
    raw = getattr(outputs[0], "raw", "") if outputs else ""
    return extract_tagged_answer(raw)


def refine_query(query: str, scores: QualityScores) -> str:
    """Refine a query using the failed evaluator scores as explicit feedback."""

    result = _run_agent(
        role="Legal retrieval query editor",
        goal="Improve document retrieval without changing the user's legal intent.",
        description=(
            "Rewrite {topic} as one concise retrieval query. Evaluator feedback: "
            "{feedback}. Preserve jurisdiction, dates, entities, and legal issue. "
            "Do not answer the question. Return only <answer>query</answer>."
        ),
        topic=query,
        feedback=scores.as_text(),
    )
    return result or query


def generate_ungrounded_answer(query: str) -> str:
    """Optional model-only fallback; the application disables it by default."""

    result = _run_agent(
        role="General legal information assistant",
        goal="Provide cautious general information without pretending it is legal advice.",
        description=(
            "Provide no more than three cautious sentences about {topic}. State that "
            "the response is not grounded in uploaded documents and depends on "
            "jurisdiction. Return only <answer>text</answer>."
        ),
        topic=query,
        feedback="No adequate document evidence was retrieved.",
    )
    return result or "No model-only answer was produced."
