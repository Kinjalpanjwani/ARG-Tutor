from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, Field

from app.core.constants import AttentionState, MoodState


class VisionStateUpdate(BaseModel):
    attention: AttentionState = AttentionState.ATTENTIVE
    mood: MoodState = MoodState.NEUTRAL


class VisionState(BaseModel):
    attention: AttentionState = AttentionState.ATTENTIVE
    mood: MoodState = MoodState.NEUTRAL
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class VisionStatusResponse(BaseModel):
    session_id: str
    attention: AttentionState
    mood: MoodState
    paused: bool = False
    cooldown: int = 0
    awaiting_checkin_reply: bool = False

