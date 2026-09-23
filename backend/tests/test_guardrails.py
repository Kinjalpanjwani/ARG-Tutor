import pytest

from app.llm.guardrails import AcademicGuard


class ClassifierLLM:
    async def json(self, messages, temperature=0):
        query = messages[-1]["content"].lower()
        return {"allowed": "restaurant" not in query, "reason": "semantic result"}


@pytest.mark.asyncio
async def test_guard_uses_semantic_classifier() -> None:
    guard = AcademicGuard(ClassifierLLM())
    assert (await guard.classify("Explain polymorphism")).allowed
    assert not (await guard.classify("Recommend a restaurant")).allowed


@pytest.mark.asyncio
async def test_guard_deterministically_allows_attached_math_worksheet() -> None:
    class LLMShouldNotRun:
        async def json(self, *args, **kwargs):
            raise AssertionError("obvious academic request should not need model classification")

    decision = await AcademicGuard(LLMShouldNotRun()).classify(
        "help solve the equations in the pdf"
    )
    assert decision.allowed
