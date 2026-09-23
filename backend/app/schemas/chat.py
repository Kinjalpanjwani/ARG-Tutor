from typing import Literal

from pydantic import BaseModel, Field

from app.core.constants import StudentLevel
from app.schemas.common import Source


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=5000)
    course_id: str | None = None
    student_level: StudentLevel = StudentLevel.SECONDARY
    language: str = Field(default="English", pattern="^(English|Urdu|Roman Urdu)$")


class ChatResponse(BaseModel):
    content: str
    allowed: bool
    source_type: Literal["course_material", "general_knowledge"]
    sources: list[Source] = Field(default_factory=list)
