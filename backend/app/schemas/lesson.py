from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.common import Source

BlockType = Literal[
    "heading", "text", "equation", "note", "example",
    "graph", "diagram", "source_image", "table",
    "student_question", "tutor_answer", "continuation",
    "step_flow", "comparison", "number_line",
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


class LessonSectionResponse(BaseModel):
    chunks: list[WhiteboardBlock]
    follow_up_questions: list[str] = Field(default_factory=list)
    has_more_sections: bool


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


class QuizGradeRequest(BaseModel):
    question: LessonQuizQuestion
    answer: str = Field(min_length=1, max_length=2000)


class QuizGradeResponse(BaseModel):
    result: Literal["correct", "partially_correct", "incorrect"]
    explanation: str
