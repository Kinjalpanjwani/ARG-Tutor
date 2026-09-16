from dataclasses import dataclass
import re

from app.llm.groq_client import GroqService
from app.llm.prompts import ACADEMIC_CLASSIFIER


@dataclass(frozen=True)
class GuardDecision:
    allowed: bool
    reason: str


class AcademicGuard:
    """Semantic, model-based domain guard with a tiny deterministic safety net."""

    def __init__(self, llm: GroqService) -> None:
        self.llm = llm

    async def classify(self, text: str) -> GuardDecision:
        normalized = text.casefold()
        academic_signals = (
            "equation", "worksheet", "textbook", "chapter", "homework", "solve",
            "math", "algebra", "geometry", "calculus", "fraction", "physics",
            "chemistry", "biology", "science", "algorithm", "programming",
            "computer science", "pdf", "slides", "question", "original page", "source page",
        )
        math_expression = bool(re.search(r"(?:\d|[a-z])\s*(?:\^|²|=|\+|−|-|×|\*)", normalized))
        if math_expression or any(signal in normalized for signal in academic_signals):
            return GuardDecision(True, "Recognized educational request")
        payload = await self.llm.json(
            [
                {"role": "system", "content": ACADEMIC_CLASSIFIER},
                {"role": "user", "content": text},
            ],
            temperature=0,
        )
        return GuardDecision(bool(payload.get("allowed", False)), str(payload.get("reason", "")))
