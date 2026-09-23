import hashlib

import numpy as np

from app.llm.guardrails import GuardDecision


class FakeEmbeddings:
    dimension = 4

    def _one(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.lower().encode()).digest()
        vector = np.array([digest[i] for i in range(self.dimension)], dtype="float32")
        return vector / np.linalg.norm(vector)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.array([self._one(text) for text in texts], dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        return self._one(text)


class FakeGuard:
    async def classify(self, text: str) -> GuardDecision:
        return GuardDecision("mattress" not in text.lower(), "academic classification")


class FakePlanner:
    async def create_plan(self, topic, student_level, language, context):
        from app.schemas.teaching import PlanStep, TeachingPlan

        return TeachingPlan(
            topic=topic,
            learning_objective=f"Understand {topic}",
            steps=[
                PlanStep(id=1, type="concept", goal="core idea"),
                PlanStep(id=2, type="check", goal="check understanding"),
            ],
        )


class FakeTutor:
    async def teach_step(self, **kwargs):
        return {"content": f"Teaching {kwargs['goal']}", "whiteboard": [], "expects_student_response": kwargs["step_type"] == "check"}

    async def interruption(self, **kwargs):
        return {"content": "Brief answer", "next_action": "repeat_simpler", "unclear_concept": "core idea", "whiteboard": []}

    async def evaluate_feedback(self, **kwargs):
        return {"content": "Good attempt", "next_action": "continue_next_step", "unclear_concept": None, "whiteboard": []}

    async def generate_checkin(self, **kwargs):
        return "Are you following along okay, or would you like a simpler example?"

    async def generate_checkin_followup(self, **kwargs):
        return f"Thanks for telling me. Here is an easier way to think about it: {kwargs['last_chunk']}."

