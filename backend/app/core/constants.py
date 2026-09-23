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


class AttentionState(StrEnum):
    ATTENTIVE = "attentive"
    DISTRACTED = "distracted"
    DROWSY = "drowsy"
    NO_FACE = "no_face"


class MoodState(StrEnum):
    NEUTRAL = "neutral"
    CONFUSED = "confused"
    STRESSED = "stressed"
    SAD = "sad"
    HAPPY = "happy"


class InterruptType(StrEnum):
    CHECKIN = "checkin"
    NO_FACE_PAUSE = "no_face_pause"


# Vision coaching heuristics & tuning constants
VISION_CHECKIN_COOLDOWN_STEPS = 3
VISION_NO_FACE_PAUSE_STREAK = 3
VISION_STATE_WINDOW = 6
VISION_PUSH_INTERVAL_SEC = 6.0
VISION_YAW_THRESHOLD_DEG = 20.0
VISION_PITCH_DOWN_THRESHOLD_DEG = 15.0
VISION_EAR_DROWSY_THRESHOLD = 0.15
# NOTE: mood/attention classification runs client-side in frontend/src/camera.js.
# These mirrors are NOT used by any backend classification today — keep them in sync
# with camera.js so they don't mislead future tuning.
VISION_CONFUSION_STREAK_NEEDED = 2
VISION_BROW_DOWN_CONFUSED_THRESHOLD = 0.06  # mirror of camera.js: real furrows run ~0.09-0.20
VISION_MOUTH_FROWN_THRESHOLD = 0.10
VISION_MOUTH_SMILE_MAX_THRESHOLD = 0.20
VISION_MOUTH_SMILE_HAPPY_THRESHOLD = 0.30
VISION_BROW_DOWN_STRESSED_THRESHOLD = 0.12
VISION_MOUTH_PRESS_THRESHOLD = 0.10


ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".pptx", ".txt", ".md"}
ACADEMIC_ONLY_MESSAGE = (
    "I’m designed for studying, coursework, lectures, academic questions, and learning. "
    "Please ask me an education-related question."
)

