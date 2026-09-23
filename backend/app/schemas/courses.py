from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field


class CourseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    subject: str | None = Field(default=None, max_length=100)
    description: str | None = Field(default=None, max_length=1000)


class Course(CourseCreate):
    course_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
