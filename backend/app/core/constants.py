from enum import StrEnum


class StudentLevel(StrEnum):
    KINDERGARTEN = "kindergarten"
    PRIMARY = "primary"
    SECONDARY = "secondary"
    COLLEGE = "college"
    UNIVERSITY = "university"


class SessionStatus(StrEnum):
    PLANNING = "planning"
    TEACHING = "teaching"
    WAITING = "waiting_for_student"
    ANSWERING = "answering_interruption"
    PAUSED = "paused"
    COMPLETED = "completed"


ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".pptx", ".txt", ".md"}
ACADEMIC_ONLY_MESSAGE = (
    "I’m designed for studying, coursework, lectures, academic questions, and learning. "
    "Please ask me an education-related question."
)
