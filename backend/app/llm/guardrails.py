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
        # Roman Urdu / Roman-Urdu-latent teaching & explanation helpers such that a
        # learning request is never rejected merely for being written in Roman Urdu.
        ru_teach = (
            "samjhao", "samjha", "samjhana", "samjhai", "samjhayen", "samjhawna",
            "sikhao", "sikao", "sikha", "seekhna", "sikhna", "taaleem",
            "parhao", "padhao", "parha", "padha", "parhna", "padhna", "sabaq", "sabak",
        )
        # Urdu-script learning signals (script must never be a rejection reason).
        urdu_teach = ("سمجھ", "سکھ", "پڑھ", "سبق", "سوال", "حل", "اردو", "تعلیم", "کہانی")
        # Language/learning framing that makes a "story" (or kahaani) request
        # educational rather than pure entertainment.
        lang_context = (
            "urdu", "اردو", "roman urdu", "language", "english", "lesson",
            "concept", "teach", "explain", "samjhao", "سکھ", "پڑھ", "سمجھ", "سبق",
        )
        story_words = ("story", "kahaani", "kahani", "کہانی", "moral", "fable",
                       "example", "analogy")

        math_expression = bool(re.search(r"(?:\d|[a-z])\s*(?:\^|²|=|\+|−|-|×|\*)", normalized))
        story_learning = any(w in normalized for w in story_words) and any(
            l in normalized for l in lang_context
        )
        if (
            math_expression
            or any(signal in normalized for signal in academic_signals)
            or any(s in normalized for s in ru_teach)
            or any(s in normalized for s in urdu_teach)
            or story_learning
        ):
            return GuardDecision(True, "Recognized educational request")
        payload = await self.llm.json(
            [
                {"role": "system", "content": ACADEMIC_CLASSIFIER},
                {"role": "user", "content": text},
            ],
            temperature=0,
        )
        return GuardDecision(bool(payload.get("allowed", False)), str(payload.get("reason", "")))
