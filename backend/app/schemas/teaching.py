from typing import Literal

from pydantic import BaseModel, Field

from app.core.constants import SessionStatus, StudentLevel
from app.schemas.common import Source, WhiteboardEvent


class PlanStep(BaseModel):
    id: int
    type: Literal["concept", "example", "check", "summary"]
    goal: str


class TeachingPlan(BaseModel):
    topic: str
    learning_objective: str
    steps: list[PlanStep] = Field(min_length=1)


class TeachingStartRequest(BaseModel):
    session_id: str


class StudentResponseRequest(BaseModel):
    content: str = Field(min_length=1, max_length=5000)
    is_interruption: bool = False


class TeachingResponse(BaseModel):
    session_id: str
    type: Literal["teaching_step", "interruption_answer", "lesson_complete"]
    content: str
    whiteboard: list[WhiteboardEvent] = Field(default_factory=list)
    source_type: Literal["course_material", "general_knowledge"]
    sources: list[Source] = Field(default_factory=list)
    expects_student_response: bool
    session_status: SessionStatus
    next_action: str | None = None
    interrupt_type: Literal["checkin", "no_face_pause"] | None = None
    paused: bool | None = None
    awaiting_checkin_reply: bool | None = None
    cooldown: int | None = None



class FollowupResponse(BaseModel):
    questions: list[str]


class QuizQuestion(BaseModel):
    question: str
    options: list[str] = Field(min_length=4, max_length=4)
    correct_index: int = Field(ge=0, le=3)
    explanation: str


class QuizResponse(BaseModel):
    questions: list[QuizQuestion]
