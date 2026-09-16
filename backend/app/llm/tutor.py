from typing import Any

from app.llm.groq_client import GroqService
from app.llm.prompts import FOLLOWUPS, INTERRUPTION, NORMAL_QA, QUIZ, STUDENT_FEEDBACK, TEACHING_STEP
from app.schemas.common import WhiteboardEvent
from app.schemas.teaching import QuizResponse


class TutorService:
    def __init__(self, llm: GroqService) -> None:
        self.llm = llm

    async def answer(self, question: str, level: str, language: str, context: str) -> str:
        return await self.llm.text(
            [
                {"role": "system", "content": NORMAL_QA},
                {"role": "system", "content": f"Level: {level}. Language: {language}.\nContext:\n{context or '(none)'}"},
                {"role": "user", "content": question},
            ]
        )

    async def teach_step(self, *, topic: str, goal: str, step_type: str, level: str, language: str, context: str, history: str) -> dict[str, Any]:
        return await self.llm.json(
            [
                {"role": "system", "content": TEACHING_STEP},
                {"role": "system", "content": f"Context:\n{context or '(none)'}\nRecent lesson:\n{history or '(none)'}"},
                {"role": "user", "content": f"Topic: {topic}\nStep type: {step_type}\nStep goal: {goal}\nLevel: {level}\nLanguage: {language}"},
            ]
        )

    async def interruption(self, *, question: str, level: str, language: str, current_goal: str, context: str, history: str) -> dict[str, Any]:
        return await self.llm.json(
            [
                {"role": "system", "content": INTERRUPTION},
                {"role": "system", "content": f"Current goal: {current_goal}\nContext:\n{context or '(none)'}\nLesson so far:\n{history}"},
                {"role": "user", "content": f"Level: {level}. Language: {language}. Question: {question}"},
            ]
        )

    async def evaluate_feedback(self, *, response: str, level: str, language: str, current_goal: str, history: str) -> dict[str, Any]:
        return await self.llm.json(
            [
                {"role": "system", "content": STUDENT_FEEDBACK},
                {"role": "system", "content": f"Current goal: {current_goal}\nLesson so far:\n{history}"},
                {"role": "user", "content": f"Level: {level}. Language: {language}. Student response: {response}"},
            ]
        )

    async def followups(self, lesson: str, level: str, covered: list[str], unclear: list[str], count: int) -> list[str]:
        data = await self.llm.json(
            [
                {"role": "system", "content": FOLLOWUPS},
                {"role": "user", "content": f"Level: {level}\nCovered: {covered}\nUnclear: {unclear}\nLesson:\n{lesson}\nCount: {count}"},
            ], temperature=0.5
        )
        return [str(item) for item in data.get("questions", [])][:count]

    async def quiz(self, lesson: str, context: str, count: int) -> QuizResponse:
        data = await self.llm.json(
            [
                {"role": "system", "content": QUIZ},
                {"role": "user", "content": f"Lesson:\n{lesson}\nCourse context:\n{context or '(none)'}\nQuestion count: {count}"},
            ]
        )
        result = QuizResponse.model_validate(data)
        return QuizResponse(questions=result.questions[:count])


def whiteboard_events(raw: Any, fallback: str) -> list[WhiteboardEvent]:
    if not isinstance(raw, list):
        return [WhiteboardEvent(content=fallback)]
    events: list[WhiteboardEvent] = []
    allowed_formats = {"text", "equation", "example", "question", "answer"}
    for item in raw:
        if isinstance(item, dict) and item.get("content"):
            output_format = item.get("format", "text")
            if output_format not in allowed_formats:
                output_format = "text"
            events.append(WhiteboardEvent(content=str(item["content"]), format=output_format))
    return events or [WhiteboardEvent(content=fallback)]
