from typing import Literal

from pydantic import BaseModel, Field


class Source(BaseModel):
    document_id: str
    document_name: str
    page: int
    chunk_id: str
    score: float
    visual_url: str | None = None


class WhiteboardEvent(BaseModel):
    type: Literal["whiteboard_update"] = "whiteboard_update"
    action: Literal["write", "clear"] = "write"
    content: str
    format: Literal["text", "equation", "example", "question", "answer"] = "text"


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str
