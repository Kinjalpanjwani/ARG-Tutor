from app.language.routing import language_instruction, output_language
from app.schemas.lesson import Flashcard, FlashcardResponse
from app.teaching.chunks import clean_markdown
from app.teaching.notebook_lesson import NotebookLessonStore


class FlashcardService:
    """Generates a front/back flashcard set from a finished lesson."""

    def __init__(self, llm, lesson_store: NotebookLessonStore) -> None:
        self.llm = llm
        self.lesson_store = lesson_store

    async def generate(self, session_id: str, count: int) -> FlashcardResponse:
        lesson = self.lesson_store.get(session_id)
        lesson_text = lesson.lecture_text.strip()
        if not lesson_text:
            lesson_text = " ".join(
                item.get("content", "")
                for item in lesson.history
                if str(item.get("role")) == "assistant" and str(item.get("content", "")).strip()
            ).strip() or lesson.topic
        system = (
            "Create a flashcard set only from the supplied lesson. Return JSON with a "
            "cards array. Each card needs a short front (the term or concept, one line "
            "or less) and a concise back (a clear spoken-friendly explanation of one to "
            "three sentences). Capture the key terms and main ideas across the lesson."
        )
        if lesson.language in ("Urdu", "Roman Urdu"):
            system = f"{system} {language_instruction(output_language(lesson.language))}"
        payload = await self.llm.json([
            {"role": "system", "content": system},
            {"role": "user", "content": f"Create up to {count} flashcards from this lesson:\n{lesson_text[:24000]}"},
        ])
        raw_cards = payload if isinstance(payload, list) else payload.get("cards", [])
        cards = []
        for item in raw_cards[:count]:
            front = clean_markdown(str(item.get("front", "") or item.get("term", "") or item.get("question", ""))).strip()
            back = clean_markdown(str(item.get("back", "") or item.get("answer", "") or item.get("explanation", ""))).strip()
            if not front or not back:
                continue
            cards.append(Flashcard(front=front, back=back))
        return FlashcardResponse(
            cards=cards,
            language=lesson.language if lesson.language in ("Urdu", "Roman Urdu") else None,
        )