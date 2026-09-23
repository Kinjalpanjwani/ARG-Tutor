from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field

from app.core.constants import AttentionState, MoodState, SessionStatus, StudentLevel
from app.schemas.common import Message, Source
from app.schemas.teaching import TeachingPlan


class SessionCreate(BaseModel):
    course_id: str | None = None
    topic: str = Field(min_length=1, max_length=300)
    student_level: StudentLevel = StudentLevel.SECONDARY
    language: str = Field(default="English", pattern="^(English|Urdu|Roman Urdu)$")


class TeachingSession(SessionCreate):
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    lesson_plan: TeachingPlan | None = None
    current_step: int = 0
    covered_concepts: list[str] = Field(default_factory=list)
    unclear_concepts: list[str] = Field(default_factory=list)
    conversation_history: list[Message] = Field(default_factory=list)
    retrieved_sources: list[Source] = Field(default_factory=list)
    taught_content: list[str] = Field(default_factory=list)
    status: SessionStatus = SessionStatus.PLANNING
    response_revision: int = 0
    latest_attention: AttentionState = AttentionState.ATTENTIVE
    latest_mood: MoodState = MoodState.NEUTRAL
    vision_history: list[dict[str, str]] = Field(default_factory=list)
    no_face_streak: int = 0
    checkin_cooldown: int = 0
    awaiting_checkin_reply: bool = False
    last_checkin_question: str | None = None
    last_checkin_directive: str | None = None
    paused_for_no_face: bool = False
    resumed_from_no_face: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

