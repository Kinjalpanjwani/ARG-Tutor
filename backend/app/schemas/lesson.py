from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.common import Source

BlockType = Literal[
    "heading", "text", "equation", "note", "example",
    "graph", "diagram", "source_image", "table",
    "student_question", "tutor_answer", "continuation",
    "step_flow", "comparison", "number_line",
    "checkin", "no_face_pause",
]


class WhiteboardBlock(BaseModel):
    id: str
    type: BlockType
    text: str
    status: Literal["pending", "active", "complete"] = "pending"
    latex: str | None = None
    source_url: str | None = None
    source_page: int | None = None
    source_document_id: str | None = None
    source_filename: str | None = None
    interrupt_type: Literal["checkin", "no_face_pause"] | None = None


class LectureStartRequest(BaseModel):
    course_id: str
    topic: str = Field(min_length=1, max_length=300)
    language: str = Field(default="English", pattern="^(English|Urdu|Roman Urdu)$")


class LectureStartResponse(BaseModel):
    session_id: str
    topic: str
    chunks: list[WhiteboardBlock]
    source_type: Literal["course_material", "general_knowledge"]
    sources: list[Source]
    intent: str
    depth: Literal["brief", "standard", "detailed", "exhaustive"] = "standard"
    follow_up_questions: list[str] = Field(default_factory=list)
    has_more_sections: bool = False
    response_type: str = "lesson"
    interrupt_type: Literal["checkin", "no_face_pause"] | None = None
    language: str | None = None


class LessonSectionResponse(BaseModel):
    chunks: list[WhiteboardBlock]
    follow_up_questions: list[str] = Field(default_factory=list)
    has_more_sections: bool
    interrupt_type: Literal["checkin", "no_face_pause"] | None = None
    language: str | None = None


class InterruptionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=5000)
    current_chunk_index: int = Field(ge=0)


class InterruptionResponse(BaseModel):
    question: WhiteboardBlock
    answer: WhiteboardBlock
    continuation: WhiteboardBlock
    resume_chunk_index: int
    source_type: Literal["course_material", "general_knowledge"]
    sources: list[Source]
    interrupt_type: Literal["checkin", "no_face_pause"] | None = None
    language: str | None = None
    answer_language: str | None = None
    reteach: bool = False



class TranscriptionResponse(BaseModel):
    speech_detected: bool
    text: str


class LessonQuizQuestion(BaseModel):
    id: str
    type: Literal["multiple_choice", "true_false", "short_answer"]
    question: str
    options: list[str] = Field(default_factory=list)


class LessonQuizResponse(BaseModel):
    questions: list[LessonQuizQuestion]
    language: str | None = None


class QuizGradeRequest(BaseModel):
    question: LessonQuizQuestion
    answer: str = Field(min_length=1, max_length=2000)


class QuizGradeResponse(BaseModel):
    result: Literal["correct", "partially_correct", "incorrect"]
    explanation: str
    language: str | None = None


class ArchiveBlock(BaseModel):
    type: BlockType
    text: str
    interrupt_type: Literal["checkin", "no_face_pause"] | None = None


class LessonArchivePayload(BaseModel):
    blocks: list[ArchiveBlock] = Field(default_factory=list)


class QuizResultEntry(BaseModel):
    question: str
    answer: str | None = None
    result: Literal["correct", "partially_correct", "incorrect"]
    explanation: str | None = None


class QuizResultsPayload(BaseModel):
    score: int = Field(ge=0)
    total: int = Field(ge=1)
    results: list[QuizResultEntry] = Field(default_factory=list)


class LessonRecord(BaseModel):
    session_id: str
    course_id: str
    topic: str
    language: str | None = None
    created_at: datetime
    updated_at: datetime
    blocks: list[ArchiveBlock] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    quiz_score: int | None = None
    quiz_total: int | None = None
    quiz_results: list[QuizResultEntry] = Field(default_factory=list)
    quiz_at: datetime | None = None


class LessonHistoryEntry(BaseModel):
    session_id: str
    topic: str
    language: str | None = None
    created_at: datetime
    updated_at: datetime
    block_count: int
    quiz_score: int | None = None
    quiz_total: int | None = None


class Flashcard(BaseModel):
    front: str
    back: str


class FlashcardResponse(BaseModel):
    cards: list[Flashcard] = Field(default_factory=list)
    language: str | None = None
