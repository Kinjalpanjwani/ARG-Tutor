from dataclasses import dataclass
import re

from app.rag.retriever import RetrievalResult


@dataclass(frozen=True)
class TutorRoute:
    intent: str
    use_uploaded_material: bool


def route_request(query: str, retrieval: RetrievalResult) -> TutorRoute:
    normalized = query.strip().casefold()
    equation = bool(re.search(r"(?:[a-z]\s*[=+\-*/^]|\d\s*[=+\-*/^])", normalized))
    explain_attachment = normalized in {"explain this", "what is this", "teach me this"}
    slide = bool(re.search(r"\bslide\s+\d+\b", normalized))
    if equation:
        intent = "solve_equation"
    elif slide:
        intent = "explain_visual"
    elif explain_attachment:
        intent = "explain_attachment"
    elif retrieval.records:
        intent = "uploaded_material_topic"
    else:
        intent = "general_topic"
    return TutorRoute(intent=intent, use_uploaded_material=bool(retrieval.records))
