import json
import re
from typing import Any

from app.core.config import get_settings
from app.llm.groq_client import GroqService
from app.llm.prompts import FOLLOWUPS, INTERRUPTION, NORMAL_QA, QUIZ, STUDENT_FEEDBACK, TEACHING_STEP
from app.schemas.common import WhiteboardEvent
from app.schemas.teaching import QuizResponse


class TutorService:
    def __init__(self, llm: GroqService) -> None:
        self.llm = llm
        self.settings = get_settings()

    async def answer(self, question: str, level: str, language: str, context: str, tone_directive: str | None = None) -> str:
        messages = [
            {"role": "system", "content": NORMAL_QA},
            {"role": "system", "content": f"Level: {level}. Language: {language}.\nContext:\n{context or '(none)'}"},
        ]
        if tone_directive:
            messages.append({"role": "system", "content": f"Tone guidance: {tone_directive}"})
        messages.append({"role": "user", "content": question})
        # Use lightweight model for normal QA
        return await self.llm.safe_call(messages, temperature=0.4, model=self.settings.light_groq_model)

    async def teach_step(self, *, topic: str, goal: str, step_type: str, level: str, language: str, context: str, history: str, lesson_state: str = "", tone_directive: str | None = None) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": TEACHING_STEP},
            {"role": "system", "content": f"Course context for this step:\n{context or '(none)'}\n\nActive lesson state:\n{lesson_state or '(none)'}\n\nRecent student/tutor dialogue:\n{history or '(none)'}"},
        ]
        if tone_directive:
            messages.append({"role": "system", "content": f"Tone guidance: {tone_directive}"})
        messages.append({"role": "user", "content": f"Topic: {topic}\nStep type: {step_type}\nStep goal: {goal}\nLevel: {level}\nLanguage: {language}"})
        raw = await self.llm.safe_call(messages, temperature=0.2, model=self.settings.light_groq_model)
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I | re.S)
        return json.loads(cleaned)

    async def interruption(self, *, question: str, level: str, language: str, current_goal: str, context: str, history: str, tone_directive: str | None = None) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": INTERRUPTION},
            {"role": "system", "content": f"Current goal: {current_goal}\nContext:\n{context or '(none)'}\nLesson so far:\n{history}"},
        ]
        if tone_directive:
            messages.append({"role": "system", "content": f"Tone guidance: {tone_directive}"})
        messages.append({"role": "user", "content": f"Level: {level}. Language: {language}. Question: {question}"})
        raw = await self.llm.safe_call(messages, temperature=0.2, model=self.settings.light_groq_model)
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I | re.S)
        return json.loads(cleaned)

    async def evaluate_feedback(self, *, response: str, level: str, language: str, current_goal: str, history: str, tone_directive: str | None = None) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": STUDENT_FEEDBACK},
            {"role": "system", "content": f"Current goal: {current_goal}\nLesson so far:\n{history}"},
        ]
        if tone_directive:
            messages.append({"role": "system", "content": f"Tone guidance: {tone_directive}"})
        messages.append({"role": "user", "content": f"Level: {level}. Language: {language}. Student response: {response}"})
        raw = await self.llm.safe_call(messages, temperature=0.2, model=self.settings.light_groq_model)
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I | re.S)
        return json.loads(cleaned)

    async def generate_checkin(self, *, directive: str, level: str, language: str) -> str:
        prompt = (
            f"{directive} Based on this, ask the student ONE short, warm, in-character "
            "check-in question that they can actually answer -- offer a specific concrete "
            "fix (a simpler example, a slower pace, trying it a different way) and ask if "
            "they want that. Do not restate the lecture content. Just the one spoken "
            "question, no quotes."
        )
        messages = [
            {"role": "system", "content": f"You are a warm, clear, patient academic tutor. Language: {language}. Level: {level}."},
            {"role": "user", "content": prompt},
        ]
        response = await self.llm.safe_call(messages, temperature=0.6, model=self.settings.light_groq_model)
        return response.strip().strip('"')

    async def generate_checkin_followup(self, *, question: str, directive: str, reply: str, last_chunk: str, level: str, language: str, history: str) -> str:
        prompt = (
            f'You just asked the student: "{question}" -- because {directive.strip()} '
            f'They replied: "{reply}". '
            f'Here is the point you were just teaching, to adapt: "{last_chunk}". '
            'Now: briefly acknowledge their reply, then actually deliver the adjusted '
            'version of that point (a simpler example if they wanted one, a slower or '
            "gentler explanation if that's what they wanted, or just move on warmly if "
            "they said they're fine) -- don't just say you'll do it, do it. End with one "
            'short natural sentence transitioning back into the lecture (e.g. moving on '
            'to the next point). This should read as ONE flowing spoken message, not '
            'separate parts. Always reply in English, no matter what language or script '
            'the student used -- do not switch languages or mirror theirs.'
        )
        messages = [
            {"role": "system", "content": f"You are a warm, clear, patient academic tutor. Level: {level}."},
            {"role": "system", "content": f"Lesson history:\n{history}"},
            {"role": "user", "content": prompt},
        ]
        response = await self.llm.safe_call(messages, temperature=0.5, model=self.settings.light_groq_model)
        return response.strip()

    async def followups(self, lesson: str, level: str, covered: list[str], unclear: list[str], count: int, tone_directive: str | None = None) -> list[str]:
        messages = [{"role": "system", "content": FOLLOWUPS}]
        if tone_directive:
            messages.append({"role": "system", "content": f"Tone guidance: {tone_directive}"})
        messages.append({"role": "user", "content": f"Level: {level}\nCovered: {covered}\nUnclear: {unclear}\nLesson:\n{lesson}\nCount: {count}"})
        raw = await self.llm.safe_call(messages, temperature=0.5, model=self.settings.light_groq_model)
        data = json.loads(raw)
        return [str(item) for item in data.get("questions", [])][:count]

    async def quiz(self, lesson: str, context: str, count: int) -> QuizResponse:
        messages = [
            {"role": "system", "content": QUIZ},
            {"role": "user", "content": f"Lesson:\n{lesson}\nCourse context:\n{context or '(none)'}\nQuestion count: {count}"},
        ]
        raw = await self.llm.safe_call(messages, temperature=0.5, model=self.settings.light_groq_model)
        data = json.loads(raw)
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
