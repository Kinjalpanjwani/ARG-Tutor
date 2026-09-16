from app.llm.groq_client import GroqService
from app.llm.prompts import PLANNER
from app.schemas.teaching import TeachingPlan


class TeachingPlanner:
    def __init__(self, llm: GroqService) -> None:
        self.llm = llm

    async def create_plan(
        self, topic: str, student_level: str, language: str, context: str = ""
    ) -> TeachingPlan:
        data = await self.llm.json(
            [
                {"role": "system", "content": PLANNER},
                {
                    "role": "user",
                    "content": (
                        f"Topic: {topic}\nStudent level: {student_level}\nLanguage: {language}\n"
                        f"Course context:\n{context or '(none; use general academic knowledge)'}"
                    ),
                },
            ]
        )
        return TeachingPlan.model_validate(data)
