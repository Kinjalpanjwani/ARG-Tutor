import asyncio
from dataclasses import dataclass, field
from threading import RLock
from uuid import uuid4
import re
import logging
from time import perf_counter

from app.core.constants import (
    ACADEMIC_ONLY_MESSAGE,
    AttentionState,
    MoodState,
    VISION_CHECKIN_COOLDOWN_STEPS,
    VISION_NO_FACE_PAUSE_STREAK,
)
from app.language.detector import SWITCH_CONFIDENCE_MIN, LanguageDetector
from app.language.routing import language_instruction, output_language, roman_urdu_instruction
from app.llm.guardrails import AcademicGuard
from app.llm.groq_client import GroqService
from app.llm.prompts import LANGUAGE_RULE
from app.rag.retriever import Retriever
from app.schemas.common import Source
from app.schemas.lesson import (
    InterruptionResponse,
    LectureStartResponse,
    WhiteboardBlock,
    LessonQuizQuestion,
    LessonQuizResponse,
    LessonSectionResponse,
    QuizGradeResponse,
)
from app.schemas.vision import VisionStateUpdate
from app.teaching.chunks import (
    BLOCK_SPLIT_LIMIT,
    clean_markdown,
    clean_spoken_text,
    lecture_blocks,
    split_prose_for_whiteboard,
)
from app.teaching.router import route_request
from app.teaching.vision import (
    build_checkin_followup_prompt,
    build_checkin_prompt,
    build_content_checkin_followup_prompt,
    tone_directive_for_state,
)


LECTURE_PERSONA = """
You are a warm, clear, patient teacher speaking aloud in a classroom. Explain rather
than merely summarize. Teach the supplied material thoroughly: unpack difficult ideas,
define important terms, explain why details matter, connect each idea to what came before,
and use concrete examples or analogies when they genuinely make the idea clearer. Do not
skip source details that are necessary to understand the section. Adapt the teaching form
to the material: use worked steps and precise notation for technical subjects; causes,
consequences, chronology, and competing perspectives for history; and characters,
motivation, setting, tension, and themes for stories. Make narrative subjects engaging,
not a dry list of facts. Preserve important source details and visual references when
useful. Never invent facts attributed to the source. If supplementing source material,
clearly label it as general background. Return valid JSON only with lesson_title,
sections (each with heading and blocks), and 2-4 follow_up_questions. Each block has
type (text, note, example, equation, graph, diagram, table, step_flow, comparison,
number_line, or checkin), content, and optional latex. For mathematics, prefer precise
equation, step_flow, number_line, graph, or diagram blocks. Do not request a source
screenshot unless the student asks to inspect the original source. Keep blocks
whiteboard-sized. Teach like a live tutor: at most one checkin block per section,
placed mid-section only after enough material has been taught to answer it (never as
the first block). A checkin block asks the student one short comprehension question
drawn from the material just taught (e.g. a value to work out, or a definition to
recall) and then pauses for their spoken answer before the lesson continues.
Always reply only in English, even if the student's request is written in Urdu or
Roman Urdu -- never mirror their language.
""" + LANGUAGE_RULE

logger = logging.getLogger(__name__)


_GRADE_AFFIRM = (
    re.compile(
        r"correctly|\b(is|was|are|were)\b[^.!?]*\b(correct|right)\b"
        r"|well\s+done|good\s+job|that's\s+(correct|right)|got\s+it|exactly|accurate|perfect|matches the expected",
        re.IGNORECASE,
    ),
    re.compile(r"صحیح|درست|ٹھیک|بالکل"),
)
_GRADE_NEGATE = (
    re.compile(
        r"incorrect|not\s+correct|isn't\s+correct|not\s+right|wrong|made\s+a\s+mistake|"
        r"failed\s+to|missing the|missing\b|doesn't\s+equal|should\s+be\b",
        re.IGNORECASE,
    ),
    re.compile(r"غلط|درست\s+نہیں|صحیح\s+نہیں"),
    re.compile(r"galat|sahi\s+nahi", re.IGNORECASE),
)


def _reconcile_grade_result(result: str, explanation: str) -> str | None:
    """Reconcile the LLM's result against its own explanation.

    The grader can emit a contradictory verdict (e.g. result 'incorrect' while the
    explanation explicitly says the answer is correct). When the explanation is
    clearly single-sided, trust it; otherwise leave the LLM's verdict alone.
    """
    aff = any(pattern.search(explanation) for pattern in _GRADE_AFFIRM)
    neg = any(pattern.search(explanation) for pattern in _GRADE_NEGATE)
    if aff and not neg:
        return "correct"
    if neg and not aff:
        return "incorrect"
    return None


INTERRUPTION_PROMPT = """
The student interrupted during a lecture. Answer directly and briefly with exactly
2-3 short bullet points, each on its own line starting with "1. ", "2. ", or "3. ".
Each point must be one spoken sentence of at most 20 words, like a teacher pausing
to clarify. No paragraphs, no headings. Use the course context as the primary
source when present. Do not restart or regenerate the lecture. Do not use Markdown.
Always reply only in English, even if the student's question is written in Urdu or
Roman Urdu -- never mirror their language.
""" + LANGUAGE_RULE

# Urdu/Roman Urdu variants of the personas above. The English-active path keeps
# using the LECTURE_PERSONA / INTERRUPTION_PROMPT constants verbatim, so English
# prompt construction stays byte-identical to the pre-bilingual behavior.
LECTURE_PERSONA_URDU = """
You are a warm, clear, patient teacher speaking aloud in a classroom. Explain rather
than merely summarize. Teach the supplied material thoroughly: unpack difficult ideas,
define important terms, explain why details matter, connect each idea to what came before,
and use concrete examples or analogies when they genuinely make the idea clearer. Do not
skip source details that are necessary to understand the section. Adapt the teaching form
to the material: use worked steps and precise notation for technical subjects; causes,
consequences, chronology, and competing perspectives for history; and characters,
motivation, setting, tension, and themes for stories. Make narrative subjects engaging,
not a dry list of facts. Preserve important source details and visual references when
useful. Never invent facts attributed to the source. If supplementing source material,
clearly label it as general background. Return valid JSON only with lesson_title,
sections (each with heading and blocks), and 2-4 follow_up_questions. Each block has
type (text, note, example, equation, graph, diagram, table, step_flow, comparison,
number_line, or checkin), content, and optional latex. For mathematics, prefer precise
equation, step_flow, number_line, graph, or diagram blocks. Do not request a source
screenshot unless the student asks to inspect the original source. Keep blocks
whiteboard-sized. Teach like a live tutor: at most one checkin block per section,
placed mid-section only after enough material has been taught to answer it (never as
the first block). A checkin block asks the student one short comprehension question
drawn from the material just taught (e.g. a value to work out, or a definition to
recall) and then pauses for their spoken answer before the lesson continues.
""" + roman_urdu_instruction("Urdu")

INTERRUPTION_PROMPT_URDU = """
The student interrupted during a lecture. Answer directly and briefly with exactly
2-3 short bullet points, each on its own line starting with "1. ", "2. ", or "3. ".
Each point must be one spoken sentence of at most 20 words, like a teacher pausing
to clarify. No paragraphs, no headings. Use the course context as the primary
source when present. Do not restart or regenerate the lecture. Do not use Markdown.
""" + roman_urdu_instruction("Urdu")

RETEACH_PROMPT = """
The student wants the current or most recently taught concept explained again. Teach
that concept again now using a genuinely different, simpler route rather than repeating
the earlier wording. Start from the student's likely point of confusion, break the idea
into small connected steps, and add one concrete example or analogy when useful. Keep
the explanation grounded in the supplied course material. Do not say the lesson has
ended and do not merely promise to explain it. Return plain spoken prose, not JSON,
headings, or Markdown.
""" + LANGUAGE_RULE


def _reteach_prompt_for(language: str) -> str:
    if language in ("Urdu", "Roman Urdu"):
        return RETEACH_PROMPT.split("Always respond only in English", 1)[0] + roman_urdu_instruction("Urdu")
    return RETEACH_PROMPT


def _lecture_persona_for(language: str) -> str:
    """Pick the lecture system persona for the active language."""
    if language in ("Urdu", "Roman Urdu"):
        return LECTURE_PERSONA_URDU
    return LECTURE_PERSONA


def _interruption_prompt_for(language: str) -> str:
    """Pick the interruption system prompt for the active language."""
    if language in ("Urdu", "Roman Urdu"):
        return INTERRUPTION_PROMPT_URDU
    return INTERRUPTION_PROMPT


def localized(language: str, english: str, urdu: str) -> str:
    """Pick hardcoded UI copy for the active language; English stays verbatim."""
    if language in ("Urdu", "Roman Urdu"):
        return urdu
    return english


_PROSE_TYPES = {"text", "example", "note"}

# Directive recorded beside a content-driven (scheduled) check-in so the reply is
# acknowledged and the lecture resumes, distinct from a vision/tone-driven check-in.
CONTENT_CHECKIN_DIRECTIVE = (
    "you paused mid-lesson to ask a comprehension check drawn from the material you "
    "just taught, and you want the student's own answer before continuing"
)


def _normalize_block_sizes(blocks: list[WhiteboardBlock]) -> list[WhiteboardBlock]:
    """Split over-length prose blocks into whiteboard-sized chunks so a single
    block never renders as one giant paragraph on the board."""
    normalized: list[WhiteboardBlock] = []
    for block in blocks:
        if block.type in _PROSE_TYPES and len(block.text) > BLOCK_SPLIT_LIMIT:
            normalized.extend(
                WhiteboardBlock(id=str(uuid4()), type=block.type, text=piece, latex=block.latex)
                for piece in split_prose_for_whiteboard(block.text)
            )
        else:
            normalized.append(block)
    return normalized


@dataclass
class NotebookLesson:
    session_id: str
    course_id: str
    topic: str
    language: str
    lecture_text: str
    chunks: list[WhiteboardBlock]
    sources: list[Source]
    history: list[dict[str, str]] = field(default_factory=list)
    follow_up_questions: list[str] = field(default_factory=list)
    pending_sections: list[tuple[str, str]] = field(default_factory=list)
    source_outline: list[str] = field(default_factory=list)
    covered_sections: list[str] = field(default_factory=list)
    active_section: str | None = None
    delivered_blocks: list[WhiteboardBlock] = field(default_factory=list)
    depth: str = "standard"
    response_type: str = "lesson"
    latest_attention: str = "attentive"
    latest_mood: str = "neutral"
    vision_history: list[dict[str, str]] = field(default_factory=list)
    no_face_streak: int = 0
    checkin_cooldown: int = 0
    awaiting_checkin_reply: bool = False
    last_checkin_question: str | None = None
    last_checkin_directive: str | None = None
    paused_for_no_face: bool = False
    resumed_from_no_face: bool = False


class NotebookLessonStore:
    def __init__(self) -> None:
        self._items: dict[str, NotebookLesson] = {}
        self._lock = RLock()

    def put(self, lesson: NotebookLesson) -> None:
        with self._lock:
            self._items[lesson.session_id] = lesson

    def get(self, session_id: str) -> NotebookLesson:
        with self._lock:
            lesson = self._items.get(session_id)
        if lesson is None:
            raise KeyError(session_id)
        return lesson

    def update_vision_state(self, session_id: str, update: VisionStateUpdate) -> NotebookLesson:
        with self._lock:
            lesson = self._items.get(session_id)
            if lesson is None:
                raise KeyError(session_id)

            lesson.latest_attention = str(update.attention)
            lesson.latest_mood = str(update.mood)
            lesson.vision_history.append({
                "attention": str(update.attention),
                "mood": str(update.mood),
            })
            if len(lesson.vision_history) > 20:
                lesson.vision_history = lesson.vision_history[-20:]

            if update.attention == AttentionState.NO_FACE:
                lesson.no_face_streak += 1
                if lesson.no_face_streak >= VISION_NO_FACE_PAUSE_STREAK:
                    lesson.paused_for_no_face = True
            else:
                lesson.no_face_streak = 0
                if lesson.paused_for_no_face:
                    lesson.paused_for_no_face = False
                    lesson.resumed_from_no_face = True
            logger.info(
                "[camera-debug] notebook_lesson=%s raw_attention=%s raw_mood=%s paused=%s cooldown=%d awaiting_reply=%s",
                session_id, update.attention, update.mood, lesson.paused_for_no_face,
                lesson.checkin_cooldown, lesson.awaiting_checkin_reply,
            )
            return lesson



class NotebookLessonService:
    """Web port of generate_lecture → split_into_chunks → indexed teach loop."""

    def __init__(
        self,
        llm: GroqService,
        guard: AcademicGuard,
        retriever: Retriever,
        language_detector: LanguageDetector | None = None,
        language_detection_enabled: bool = False,
    ) -> None:
        self.llm = llm
        self.guard = guard
        self.retriever = retriever
        self.language_detector = language_detector
        self.language_detection_enabled = language_detection_enabled
        self.store = NotebookLessonStore()
        self._checkin_locks: dict[str, asyncio.Lock] = {}

    async def _detected_language(self, text: str) -> str | None:
        """Return a confidently detected language for a student-turn text, or None
        when detection is disabled, unavailable, or the turn is inconclusive.
        The flag check happens BEFORE the detector is invoked."""
        if not self.language_detection_enabled or self.language_detector is None:
            return None
        candidate, confidence = await self.language_detector.analyze(text)
        if confidence >= SWITCH_CONFIDENCE_MIN:
            return candidate
        return None

    def response_language(self, lesson: NotebookLesson) -> str | None:
        """Advertise the active language on responses only while detection is enabled."""
        return lesson.language if self.language_detection_enabled else None

    def _checkin_lock(self, session_id: str) -> asyncio.Lock:
        lock = self._checkin_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._checkin_locks[session_id] = lock
        return lock

    @staticmethod
    def _depth(topic: str) -> str:
        value = topic.casefold()
        if any(term in value for term in ("every page", "page by page", "line by line")):
            return "exhaustive"
        if any(term in value for term in ("in detail", "deep explanation", "every chapter", "step by step")):
            return "detailed"
        if any(term in value for term in ("briefly", "short summary", "quick summary", "overview")):
            return "brief"
        return "standard"

    @staticmethod
    def _is_reteach_request(text: str) -> bool:
        value = re.sub(r"\s+", " ", text.strip().casefold())
        return any(re.search(pattern, value) for pattern in (
            r"\b(?:teach|explain|show|go over|walk (?:me )?through|repeat)\b.*\b(?:again|once more)\b",
            r"\b(?:i )?(?:did not|didn't|dont|don't|could not|couldn't) understand\b",
            r"\b(?:repeat|reteach) (?:this|that|it|the concept|the section)\b",
        ))

    @staticmethod
    def _recent_teaching_context(lesson: NotebookLesson) -> tuple[str, str]:
        teaching_types = {
            "text", "example", "equation", "graph", "diagram", "table",
            "step_flow", "comparison", "number_line", "note",
        }
        relevant = [block for block in lesson.delivered_blocks if block.type in teaching_types]
        if not relevant:
            return lesson.topic, lesson.topic
        current = relevant[-1].text
        previous = "\n".join(block.text for block in relevant[-5:])
        return current, previous

    @staticmethod
    def _intent(topic: str) -> str:
        value = topic.strip().casefold()
        if re.fullmatch(r"(?:the )?table of \d+\??", value): return "SHOW_TABLE"
        if any(term in value for term in ("just give the answer", "answer only")): return "DIRECT_ANSWER"
        if re.search(r"\bquiz (?:me|on)\b", value): return "QUIZ"
        if any(term in value for term in ("step by step", "solve ")): return "EXPLAIN_STEP_BY_STEP"
        if any(term in value for term in ("this pdf", "the pdf", "this document", "these slides")): return "EXPLAIN_DOCUMENT"
        if any(term in value for term in ("summary", "summarize", "overview")): return "SUMMARIZE"
        if any(term in value for term in ("in detail", "from scratch", "teach me")): return "DETAILED_LESSON"
        return "TEACH_TOPIC"

    @staticmethod
    def _table_fast_path(topic: str, language: str) -> tuple[str, list[WhiteboardBlock], str] | None:
        match = re.fullmatch(
            r"(?:(?:the )?table of (\d+)\??|(?:mujhe )?(\d+)\s+ka\s+(?:table|pahada|pahad|pahra|pahar|phara)\??(?:\s+batao\s*)?)",
            topic.strip(),
            re.I,
        )
        if not match: return None
        number = int(match.group(1) or match.group(2))
        if language in ("Urdu", "Roman Urdu"):
            heading = f"{number} کا پہاڑا" if language == "Urdu" else f"{number} ka pahada"
            note = "کیا آپ اس کی مشق کرنا چاہیں گے؟" if language == "Urdu" else "Kya aap is ki mashq karna chahenge?"
            follow_up = (
                f"کیا آپ {number} کا پہاڑا 10 سے آگے جاری رکھ سکتے ہیں؟"
                if language == "Urdu"
                else f"Kya aap {number} ka pahada 10 say aage jari rakh saktay hain?"
            )
        else:
            heading = f"Table of {number}"
            note = "Want to practice it?"
            follow_up = f"Can you continue the table of {number} beyond 10?"
        if language in ("Urdu", "Roman Urdu"):
            checkin = (
                f"اب آپ بتائیں: 7 × {number} کیا ہے؟"
                if language == "Urdu"
                else f"Ab aap batayein: 7 × {number} kya hai?"
            )
        else:
            checkin = f"Now you tell me: what is 7 × {number}?"
        rows = [
            WhiteboardBlock(id=str(uuid4()), type="table", text=f"{number} × {i} = {number * i}")
            for i in range(1, 11)
        ]
        blocks = (
            [WhiteboardBlock(id=str(uuid4()), type="heading", text=heading)]
            + rows[:5]
            + [WhiteboardBlock(id=str(uuid4()), type="checkin", text=checkin)]
            + rows[5:]
            + [WhiteboardBlock(id=str(uuid4()), type="note", text=note)]
        )
        return "multiplication_table", blocks, follow_up

    @staticmethod
    def _structured_blocks(payload: object) -> tuple[list[WhiteboardBlock], list[str]]:
        if not isinstance(payload, dict) or not isinstance(payload.get("sections"), list):
            raise ValueError("Invalid structured lesson")
        blocks: list[WhiteboardBlock] = []
        for section in payload["sections"]:
            if not isinstance(section, dict):
                continue
            heading = clean_markdown(str(section.get("heading", "")))
            if heading:
                blocks.append(WhiteboardBlock(id=str(uuid4()), type="heading", text=heading))
            for item in section.get("blocks", []):
                if not isinstance(item, dict):
                    continue
                raw_content = item.get("content", "")
                if isinstance(raw_content, list):
                    raw_content = "\n↓\n".join(str(part) for part in raw_content)
                content = clean_markdown(str(raw_content))
                block_type = item.get("type", "text")
                if block_type not in {"text", "note", "example", "equation", "graph", "diagram", "table", "step_flow", "comparison", "number_line", "checkin"}:
                    block_type = "text"
                if content:
                    blocks.append(WhiteboardBlock(
                        id=str(uuid4()), type=block_type, text=content,
                        latex=str(item.get("latex")) if item.get("latex") else None,
                    ))
        questions = [clean_markdown(str(q)) for q in payload.get("follow_up_questions", []) if str(q).strip()][:4]
        return blocks, questions

    def _scheduled_checkin_response(self, lesson: NotebookLesson, block: WhiteboardBlock) -> LessonSectionResponse | None:
        """Turn a content-scheduled check-in block into an interactive pause, or return
        None when the block is a normal teaching block. Reuses the same reply-awaiting
        machinery as vision-driven check-ins so the student gets the warm acknowledge →
        answer → continue experience without frontend changes."""
        if block.type != "checkin":
            return None
        lesson.awaiting_checkin_reply = True
        lesson.checkin_cooldown = VISION_CHECKIN_COOLDOWN_STEPS
        lesson.last_checkin_question = clean_spoken_text(block.text)
        lesson.last_checkin_directive = CONTENT_CHECKIN_DIRECTIVE
        lesson.history.append({"role": "assistant", "content": lesson.last_checkin_question})
        block.interrupt_type = "checkin"
        block.status = "active"
        return LessonSectionResponse(
            chunks=[block],
            follow_up_questions=lesson.follow_up_questions[:4],
            has_more_sections=bool(lesson.chunks or lesson.pending_sections),
            interrupt_type="checkin",
            language=self.response_language(lesson),
        )

    @staticmethod
    def _comprehensive_material_request(topic: str) -> bool:
        value = topic.casefold()
        return any(re.search(pattern, value) for pattern in (
            r"\b(?:all|every|each)\s+(?:the\s+)?(?:pdfs?|documents?|files?|chapters?|pages?|slides?)\b",
            r"\b(?:these|uploaded)\s+(?:pdfs?|documents?|files?|chapters?|materials?|slides?)\b",
            r"\b(?:teach|explain|cover|go through)\b.*\b(?:all|everything|them)\b",
            r"\b(?:page[ -]by[ -]page|chapter[ -]by[ -]chapter)\b",
        ))

    @staticmethod
    def _records_to_sections(
        document_name: str, records: list, split_chapters: bool, include_document_name: bool,
    ) -> list[tuple[str, str]]:
        """Create bounded, ordered source sections without dropping later material."""
        full_text = "\n\n".join(record.page_content for record in records)
        if split_chapters:
            matches = list(re.finditer(
                r"(?im)^\s*chapter\s+([\w-]+)(?:\s*[:.-]\s*([^\n]+))?\s*$", full_text
            ))
            if matches:
                return [
                    (
                        (f"{document_name} — " if include_document_name else "")
                        + f"Chapter {match.group(1)}"
                        + (f": {match.group(2).strip()}" if match.group(2) else ""),
                        full_text[match.start(): matches[index + 1].start() if index + 1 < len(matches) else len(full_text)].strip(),
                    )
                    for index, match in enumerate(matches)
                ]
            inferred = []
            for record in records:
                first = next((line.strip() for line in record.page_content.splitlines() if line.strip()), "")
                words = first.split()
                if 1 < len(words) <= 10 and not first.endswith((".", "?", "!")):
                    prefix = f"{document_name} — " if include_document_name else ""
                    inferred.append((f"{prefix}{first} (page {record.page})", record.page_content))
            if len(inferred) >= 2:
                return inferred

        # Keep each generation request comfortably below the model input limit while
        # retaining every ordered chunk from a comprehensive multi-document lesson.
        sections: list[tuple[str, str]] = []
        batch: list = []
        batch_size = 0
        for record in records:
            size = len(record.page_content)
            if batch and batch_size + size > 12000:
                page_label = f"pages {batch[0].page}-{batch[-1].page}" if batch[0].page != batch[-1].page else f"page {batch[0].page}"
                prefix = f"{document_name} — " if include_document_name else ""
                sections.append((f"{prefix}{page_label.capitalize()}", "\n\n".join(item.page_content for item in batch)))
                batch, batch_size = [], 0
            batch.append(record)
            batch_size += size
        if batch:
            page_label = f"pages {batch[0].page}-{batch[-1].page}" if batch[0].page != batch[-1].page else f"page {batch[0].page}"
            prefix = f"{document_name} — " if include_document_name else ""
            sections.append((f"{prefix}{page_label.capitalize()}", "\n\n".join(item.page_content for item in batch)))
        return sections

    def _source_sections(self, topic: str, retrieval, course_id: str | None = None) -> list[tuple[str, str]]:
        lowered = topic.casefold()
        comprehensive = self._comprehensive_material_request(topic)
        if course_id is None and retrieval.records:
            course_id = retrieval.records[0].course_id
        if comprehensive:
            records = [record for record in self.retriever.store.records if record.course_id == course_id]
        else:
            if not retrieval.records:
                return [("Requested topic", "")]
            retrieved_ids = {record.document_id for record in retrieval.records}
            # A request for a particular chapter/document needs the complete matching
            # source, while an ordinary topic lesson should stay on the retrieved chunks.
            wants_complete_source = any(term in lowered for term in ("chapter", "this pdf", "the pdf", "this document", "the document"))
            records = [
                record for record in self.retriever.store.records
                if record.course_id == course_id and record.document_id in retrieved_ids
                and (wants_complete_source or record.chunk_id in {item.chunk_id for item in retrieval.records})
            ]
        if not records:
            records = list(retrieval.records)

        grouped: dict[str, list] = {}
        for record in records:
            grouped.setdefault(record.document_id, []).append(record)
        for items in grouped.values():
            items.sort(key=lambda record: (record.page, record.chunk_id))

        if "every page" in lowered or "page by page" in lowered:
            requested_chapter = re.search(r"chapter\s+([\w-]+)", topic, re.I)
            sections: list[tuple[str, str]] = []
            for items in grouped.values():
                selected = items
                if requested_chapter:
                    token = requested_chapter.group(1)
                    start = next((i for i, record in enumerate(items) if re.search(rf"\bchapter\s+{re.escape(token)}\b", record.page_content, re.I)), None)
                    if start is not None:
                        end = next((i for i in range(start + 1, len(items)) if re.search(r"\bchapter\s+[\w-]+", items[i].page_content, re.I)), len(items))
                        selected = items[start:end]
                pages: dict[int, list[str]] = {}
                for record in selected:
                    pages.setdefault(record.page, []).append(record.page_content)
                name = selected[0].document_name if selected else items[0].document_name
                prefix = f"{name} — " if len(grouped) > 1 else ""
                sections.extend((f"{prefix}Page {page}", "\n".join(text)) for page, text in pages.items())
            return sections

        split_chapters = "chapter" in lowered or comprehensive
        sections = []
        for items in grouped.values():
            sections.extend(self._records_to_sections(
                items[0].document_name, items, split_chapters, include_document_name=len(grouped) > 1
            ))
        return sections or [("Requested topic", retrieval.context)]

    async def _generate_section(
        self, topic: str, language: str, depth: str, title: str, context: str,
        tone_directive: str | None = None, lesson_state: str = "",
    ) -> tuple[list[WhiteboardBlock], list[str]]:
        depth_rules = {
            "brief": "Use 3-5 concise teaching blocks.",
            "standard": "Use 9-15 substantial teaching blocks, including useful examples and one natural comprehension check.",
            "detailed": "Use 14-22 substantial teaching blocks with important details, connections, examples, and periodic interaction.",
            "exhaustive": "Explain this source section thoroughly in 15-25 teaching blocks, preserving sequence and details.",
        }
        messages = [
            {"role": "system", "content": _lecture_persona_for(language)},
            {"role": "system", "content": f"Follow the student's explicit request first; do not add generic importance/history unless asked. Depth: {depth}. {depth_rules[depth]}\nCurrent source section: {title}\n\nACTIVE LESSON STATE:\n{lesson_state or '(This is the first section.)'}\n\nGROUNDING MATERIAL FOR THIS SECTION:\n{context or '(No relevant attachment; use general academic knowledge.)'}\n\nCover the grounding material for this section before moving on. Maintain continuity with the lesson state, avoid repeating already-taught explanations, and never claim a detail is in the source unless it appears in the grounding material."},
        ]
        if tone_directive:
            messages.append({"role": "system", "content": f"Tone guidance: {tone_directive}"})
        messages.append({"role": "user", "content": f"Student request: {topic}\nTeach section: {title}\nLanguage: {language}"})
        try:
            blocks, questions = self._structured_blocks(await self.llm.json(messages, temperature=0.45))
            return _normalize_block_sizes(blocks), questions
        except Exception:
            fallback = clean_spoken_text(await self.llm.text(messages + [{"role": "system", "content": "JSON failed. Return a plain spoken lesson now."}], temperature=0.45))
            return _normalize_block_sizes(lecture_blocks(fallback)), []

    @staticmethod
    def _lesson_state(lesson: NotebookLesson, next_title: str) -> str:
        """Build a compact, stable memory packet for the next generation call."""
        taught = "\n".join(
            f"- {block.text}" for block in lesson.delivered_blocks[-24:]
            if block.type not in {"source_image", "no_face_pause"}
        )
        if len(taught) > 7000:
            taught = taught[-7000:]
        dialogue = "\n".join(
            f"{item.get('role', 'unknown')}: {item.get('content', '')}"
            for item in lesson.history[-10:]
            if str(item.get("content", "")).strip()
        )
        if len(dialogue) > 4000:
            dialogue = dialogue[-4000:]
        remaining = [title for title, _ in lesson.pending_sections]
        return (
            f"Lesson topic/request: {lesson.topic}\n"
            f"Full source-section outline: {lesson.source_outline}\n"
            f"Sections already taught: {lesson.covered_sections}\n"
            f"Section just completed: {lesson.active_section or '(none)'}\n"
            f"Section to teach now: {next_title}\n"
            f"Sections still remaining after this one: {remaining}\n"
            f"Recent explanations actually delivered to the student:\n{taught or '(none)'}\n"
            f"Relevant recent student questions and tutor replies:\n{dialogue or '(none)'}"
        )

    @staticmethod
    def _record_delivery(lesson: NotebookLesson, block: WhiteboardBlock) -> None:
        lesson.delivered_blocks.append(block)

    async def start(self, course_id: str, topic: str, language: str) -> LectureStartResponse:
        request_start = perf_counter()
        logger.info("[TUTOR] request received topic=%r", topic)
        detected = await self._detected_language(topic)
        active_language = detected or language
        intent = self._intent(topic)
        fast = self._table_fast_path(topic, active_language)
        if fast:
            response_type, chunks, follow_up = fast
            session_id = str(uuid4())
            lecture = " ".join(block.text for block in chunks)
            self.store.put(NotebookLesson(session_id, course_id, topic, active_language, lecture, chunks, [], history=[{"role":"assistant","content":lecture}], follow_up_questions=[follow_up], response_type=response_type))
            logger.info("[WHITEBOARD] blocks received=%d [PERF] generation: 0ms", len(chunks))
            return LectureStartResponse(session_id=session_id, topic=topic, chunks=chunks, source_type="general_knowledge", sources=[], intent=intent, depth="brief", follow_up_questions=self.store.get(session_id).follow_up_questions, response_type=response_type, language=self.response_language(self.store.get(session_id)))
        decision = await self.guard.classify(topic)
        if not decision.allowed:
            raise ValueError(ACADEMIC_ONLY_MESSAGE)
        logger.info("[INTENT] %s [PERF] intent: %.0fms", intent, (perf_counter() - request_start) * 1000)
        retrieval_start = perf_counter()
        retrieval = self.retriever.retrieve(topic, course_id)
        logger.info("[PERF] retrieval: %.0fms", (perf_counter() - retrieval_start) * 1000)
        route = route_request(topic, retrieval)
        depth = self._depth(topic)

        # Generate first source section
        source_sections = self._source_sections(topic, retrieval, course_id)
        title, context = source_sections[0]
        generation_start = perf_counter()
        logger.info("[LLM] request start section=%r", title)
        first_state = (
            f"Lesson topic/request: {topic}\n"
            f"Full source-section outline: {[section_title for section_title, _ in source_sections]}\n"
            f"Sections already taught: []\nSection to teach now: {title}"
        )
        section_blocks, section_questions = await self._generate_section(
            topic, active_language, depth, title, context[:24000], lesson_state=first_state
        )
        logger.info("[LLM] request success [PERF] generation: %.0fms [WHITEBOARD] blocks received=%d", (perf_counter() - generation_start) * 1000, len(section_blocks))

        # Visual source blocks
        visual_sources: list[WhiteboardBlock] = []
        seen_visuals: set[str] = set()
        explicit_source_visual = bool(re.search(r"\b(?:show|display|inspect|look at)\b.*\b(?:original|source|page|image|diagram|figure)\b", topic, re.I))
        for source in sorted(retrieval.sources, key=lambda item: item.page) if explicit_source_visual else []:
            if source.visual_url and source.visual_url not in seen_visuals:
                seen_visuals.add(source.visual_url)
                visual_sources.append(WhiteboardBlock(
                    id=str(uuid4()), type="source_image",
                    text=f"Source visual from {source.document_name}, page {source.page}",
                    source_url=source.visual_url, source_page=source.page,
                    source_document_id=source.document_id, source_filename=source.document_name,
                ))

        # Split first block from the rest
        first_block = section_blocks[0] if section_blocks else None
        remaining_blocks = section_blocks[1:] if len(section_blocks) > 1 else []

        if first_block and first_block.type == "checkin":
            # A check-in must flow through next_section so the backend arms the
            # awaiting-reply state before the student hears the question.
            remaining_blocks.insert(0, first_block)
            first_block = None

        # Build initial chunks (visuals + first block)
        initial_chunks: list[WhiteboardBlock] = visual_sources[:2]
        if first_block:
            initial_chunks.append(first_block)

        # Store lesson with remaining blocks and pending sections
        session_id = str(uuid4())
        lesson = NotebookLesson(
            session_id=session_id,
            course_id=course_id,
            topic=topic,
            language=active_language,
            # Keep the taught content as the lesson's memory so quizzes and interrupt
            # context work even when there is no uploaded material.
            lecture_text=" ".join(block.text for block in section_blocks),
            chunks=remaining_blocks,
            sources=retrieval.sources,
            history=[],
            follow_up_questions=section_questions[:4],
            pending_sections=source_sections[1:],
            source_outline=[section_title for section_title, _ in source_sections],
            active_section=title,
            delivered_blocks=list(initial_chunks),
            depth=depth,
            response_type="lesson",
        )
        self.store.put(lesson)
        return LectureStartResponse(
            session_id=session_id,
            topic=topic,
            chunks=initial_chunks,
            source_type=retrieval.source_type,
            sources=retrieval.sources,
            intent=intent,
            depth=depth,
            follow_up_questions=section_questions[:4],
            has_more_sections=bool(source_sections[1:] or remaining_blocks),
            response_type="lesson",
            language=self.response_language(lesson),
        )

    def cancel_checkin(self, session_id: str) -> NotebookLesson:
        """Clear a pending check-in that the client could not deliver (e.g. dropped due
        to a concurrent interrupt in flight). Clears the reply-awaiting flags so the
        next student question is not misread as this check-in's reply; cooldown is left
        untouched so a fresh push re-fires naturally once it decays."""
        lesson = self.store.get(session_id)
        lesson.awaiting_checkin_reply = False
        lesson.last_checkin_question = None
        lesson.last_checkin_directive = None
        return lesson

    async def maybe_fire_checkin(self, session_id: str) -> str | None:
        """Evaluate the check-in gate on every vision push, independent of block timing.

        No-op when a check-in is pending, when the cooldown hasn't expired yet, or when
        the freshest (attention, mood) reading isn't a trigger. Decays checkin_cooldown
        on this push cadence so cooldown counts pushes, not section advances.
        """
        async with self._checkin_lock(session_id):
            lesson = self.store.get(session_id)
            if lesson.awaiting_checkin_reply:
                return None
            if lesson.checkin_cooldown > 0:
                lesson.checkin_cooldown -= 1
                return None
            tone_directive = tone_directive_for_state(lesson.latest_attention, lesson.latest_mood)
            if tone_directive is None:
                return None
            if not (
                lesson.latest_attention in {AttentionState.DISTRACTED, AttentionState.DROWSY}
                or lesson.latest_mood in {MoodState.CONFUSED, MoodState.STRESSED, MoodState.SAD}
            ):
                return None
            q_prompt = build_checkin_prompt(tone_directive, language=lesson.language)
            try:
                question = clean_spoken_text(await self.llm.text([
                    {"role": "system", "content": _lecture_persona_for(lesson.language)},
                    {"role": "user", "content": q_prompt},
                ], temperature=0.6))
            except Exception:
                question = localized(lesson.language, "Are you following along okay, or would you prefer a simpler example?", "کیا آپ سمجھ رہے ہیں؟ یا کوئی آسان مثال چاہیں گے؟")
            lesson.awaiting_checkin_reply = True
            lesson.checkin_cooldown = VISION_CHECKIN_COOLDOWN_STEPS
            lesson.last_checkin_question = question
            lesson.last_checkin_directive = tone_directive
            lesson.history.append({"role": "assistant", "content": question})
            return question

    async def next_section(self, session_id: str) -> LessonSectionResponse:
        lesson = self.store.get(session_id)

        # 1. No-face pause
        if lesson.paused_for_no_face:
            return LessonSectionResponse(
                chunks=[WhiteboardBlock(id=str(uuid4()), type="no_face_pause", text=localized(lesson.language, "Still there? Pausing until you're back.", "کیا آپ وہاں ہیں؟ آپ کے واپس آنے تک میں رکی رہوں گا۔"), status="active", interrupt_type="no_face_pause")],
                follow_up_questions=lesson.follow_up_questions[:4],
                has_more_sections=True,
                interrupt_type="no_face_pause",
                language=self.response_language(lesson),
            )

        # 2. Resumed from no-face
        if lesson.resumed_from_no_face:
            lesson.resumed_from_no_face = False
            return LessonSectionResponse(
                chunks=[WhiteboardBlock(id=str(uuid4()), type="note", text=localized(lesson.language, "Welcome back — let's pick up where we left off.", "خوش آمدید — چلیں وہیں سے جاری رکھتے ہیں جہاں رکے تھے۔"), status="active")],
                follow_up_questions=lesson.follow_up_questions[:4],
                has_more_sections=bool(lesson.chunks or lesson.pending_sections),
                language=self.response_language(lesson),
            )

        # 2b. Check-in already fired via a vision push and is awaiting a reply.
        # Return nothing rather than popping a normal chunk (which would silently
        # lose lecture content if the frontend polls next-section while awaiting).
        if lesson.awaiting_checkin_reply:
            return LessonSectionResponse(
                chunks=[],
                follow_up_questions=lesson.follow_up_questions[:4],
                has_more_sections=True,
                language=self.response_language(lesson),
            )

        tone_directive = tone_directive_for_state(lesson.latest_attention, lesson.latest_mood)

        # 3. Check-in question
        needs_checkin = (
            lesson.checkin_cooldown <= 0
            and tone_directive is not None
            and (
                lesson.latest_attention in {AttentionState.DISTRACTED, AttentionState.DROWSY}
                or lesson.latest_mood in {MoodState.CONFUSED, MoodState.STRESSED, MoodState.SAD}
            )
        )

        if needs_checkin:
            q_prompt = build_checkin_prompt(tone_directive, language=lesson.language)
            try:
                question = clean_spoken_text(await self.llm.text([
                    {"role": "system", "content": _lecture_persona_for(lesson.language)},
                    {"role": "user", "content": q_prompt},
                ], temperature=0.6))
            except Exception:
                question = localized(lesson.language, "Are you following along okay, or would you prefer a simpler example?", "کیا آپ سمجھ رہے ہیں؟ یا کوئی آسان مثال چاہیں گے؟")

            lesson.awaiting_checkin_reply = True
            lesson.checkin_cooldown = VISION_CHECKIN_COOLDOWN_STEPS
            lesson.last_checkin_question = question
            lesson.last_checkin_directive = tone_directive
            lesson.history.append({"role": "assistant", "content": question})
            return LessonSectionResponse(
                chunks=[WhiteboardBlock(id=str(uuid4()), type="checkin", text=question, status="active", interrupt_type="checkin")],
                follow_up_questions=lesson.follow_up_questions[:4],
                has_more_sections=True,
                interrupt_type="checkin",
                language=self.response_language(lesson),
            )

        lesson.checkin_cooldown = max(0, lesson.checkin_cooldown - 1)

        # 4. Check lesson.chunks first for unconsumed blocks
        if lesson.chunks:
            block = lesson.chunks.pop(0)
            self._record_delivery(lesson, block)
            checkin = self._scheduled_checkin_response(lesson, block)
            if checkin is not None:
                return checkin
            return LessonSectionResponse(
                chunks=[block],
                follow_up_questions=lesson.follow_up_questions[:4],
                has_more_sections=bool(lesson.chunks or lesson.pending_sections),
                language=self.response_language(lesson),
            )

        # 5. Only fall through to pending_sections once lesson.chunks is empty
        if not lesson.pending_sections:
            return LessonSectionResponse(chunks=[], follow_up_questions=lesson.follow_up_questions[:4], has_more_sections=False, language=self.response_language(lesson))

        title, context = lesson.pending_sections.pop(0)
        if lesson.active_section and lesson.active_section not in lesson.covered_sections:
            lesson.covered_sections.append(lesson.active_section)
        lesson_state = self._lesson_state(lesson, title)
        logger.info("[LLM] request start section=%r", title)
        started = perf_counter()
        blocks, questions = await self._generate_section(
            lesson.topic,
            lesson.language,
            lesson.depth,
            title,
            context[:24000],
            tone_directive=tone_directive,
            lesson_state=lesson_state,
        )
        lesson.active_section = title
        blocks.insert(0, WhiteboardBlock(id=str(uuid4()), type="heading", text=title))
        lesson.lecture_text += " " + " ".join(block.text for block in blocks)
        lesson.follow_up_questions = (lesson.follow_up_questions + questions)[:4]
        logger.info(
            "[LLM] request success [PERF] generation: %.0fms [WHITEBOARD] blocks received=%d",
            (perf_counter() - started) * 1000,
            len(blocks),
        )

        lesson.chunks.extend(blocks)
        first_block = lesson.chunks.pop(0) if lesson.chunks else None
        if first_block is None:
            return LessonSectionResponse(
                chunks=[],
                follow_up_questions=lesson.follow_up_questions,
                has_more_sections=bool(lesson.chunks or lesson.pending_sections),
                language=self.response_language(lesson),
            )
        self._record_delivery(lesson, first_block)
        checkin = self._scheduled_checkin_response(lesson, first_block)
        if checkin is not None:
            return checkin
        return LessonSectionResponse(
            chunks=[first_block],
            follow_up_questions=lesson.follow_up_questions,
            has_more_sections=bool(lesson.chunks or lesson.pending_sections),
            language=self.response_language(lesson),
        )

    async def interrupt(self, session_id: str, question: str, current_index: int) -> InterruptionResponse:
        lesson = self.store.get(session_id)
        current, recent_teaching = self._recent_teaching_context(lesson)

        # The interrupt is answered in the detected language, but the lesson itself
        # keeps its original language so the lecture resumes where it left off.
        detected = await self._detected_language(question)
        answer_language = detected or lesson.language

        # Handle check-in reply (vision-driven or content-scheduled)
        if lesson.awaiting_checkin_reply:
            lesson.awaiting_checkin_reply = False
            content_checkin = lesson.last_checkin_directive == CONTENT_CHECKIN_DIRECTIVE
            if content_checkin:
                prompt = build_content_checkin_followup_prompt(
                    question=lesson.last_checkin_question or "",
                    reply=question,
                    last_chunk=current,
                    language=answer_language,
                )
            else:
                directive = lesson.last_checkin_directive or "The student asked for guidance."
                prompt = build_checkin_followup_prompt(
                    question=lesson.last_checkin_question or "",
                    directive=directive,
                    reply=question,
                    last_chunk=current,
                    language=answer_language,
                )
            answer = clean_spoken_text(await self.llm.safe_call([
                {"role": "system", "content": _lecture_persona_for(answer_language)},
                *lesson.history[-6:],
                {"role": "user", "content": prompt},
            ], temperature=0.5))
            lesson.history.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ])
            resume = current_index + 1
            return InterruptionResponse(
                question=WhiteboardBlock(id=str(uuid4()), type="student_question", text=question, status="complete"),
                answer=WhiteboardBlock(id=str(uuid4()), type="tutor_answer", text=answer, status="active", interrupt_type="checkin"),
                continuation=WhiteboardBlock(id=str(uuid4()), type="continuation", text=localized(lesson.language, "Continuing the lesson…", "سبق جاری رکھ رہے ہیں…"), status="pending"),
                resume_chunk_index=resume,
                source_type="general_knowledge",
                sources=[],
                interrupt_type="checkin",
                language=self.response_language(lesson),
                answer_language=answer_language if self.language_detection_enabled else None,
            )

        # Content check-in embedded in a multiplication-table lesson, which is
        # delivered all at once: the frontend's chunk index maps directly onto
        # lesson.chunks, so an interrupt raised on a check-in block is that check-in's
        # reply rather than a free-form question.
        if (
            lesson.response_type == "multiplication_table"
            and lesson.chunks
            and current_index < len(lesson.chunks)
            and lesson.chunks[current_index].type == "checkin"
        ):
            lesson.last_checkin_question = lesson.chunks[current_index].text
            lesson.last_checkin_directive = CONTENT_CHECKIN_DIRECTIVE
            prompt = build_content_checkin_followup_prompt(
                question=lesson.last_checkin_question,
                reply=question,
                last_chunk=current,
                language=answer_language,
            )
            answer = clean_spoken_text(await self.llm.safe_call([
                {"role": "system", "content": _lecture_persona_for(answer_language)},
                *lesson.history[-6:],
                {"role": "user", "content": prompt},
            ], temperature=0.5))
            lesson.history.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ])
            resume = current_index + 1
            return InterruptionResponse(
                question=WhiteboardBlock(id=str(uuid4()), type="student_question", text=question, status="complete"),
                answer=WhiteboardBlock(id=str(uuid4()), type="tutor_answer", text=answer, status="active", interrupt_type="checkin"),
                continuation=WhiteboardBlock(id=str(uuid4()), type="continuation", text=localized(lesson.language, "Continuing the lesson…", "سبق جاری رکھ رہے ہیں…"), status="pending"),
                resume_chunk_index=resume,
                source_type="general_knowledge",
                sources=[],
                interrupt_type="checkin",
                language=self.response_language(lesson),
                answer_language=answer_language if self.language_detection_enabled else None,
            )

        if self._is_reteach_request(question):
            retrieval = self.retriever.retrieve(f"{lesson.topic}: {current}", lesson.course_id)
            messages = [
                {"role": "system", "content": _reteach_prompt_for(answer_language)},
                {
                    "role": "system",
                    "content": (
                        f"Active lesson topic: {lesson.topic}\n"
                        f"Current source section: {lesson.active_section or lesson.topic}\n"
                        f"Most recently taught concept:\n{current}\n"
                        f"Recent explanation context:\n{recent_teaching}\n"
                        f"Relevant course material:\n{retrieval.context or '(none)'}"
                    ),
                },
                *lesson.history[-6:],
                {"role": "user", "content": question},
            ]
            answer = clean_spoken_text(await self.llm.safe_call(messages, temperature=0.45))
            lesson.history.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ])
            return InterruptionResponse(
                question=WhiteboardBlock(id=str(uuid4()), type="student_question", text=question, status="complete"),
                answer=WhiteboardBlock(id=str(uuid4()), type="tutor_answer", text=answer, status="active"),
                continuation=WhiteboardBlock(id=str(uuid4()), type="continuation", text=localized(lesson.language, "Continuing the lesson…", "سبق جاری رکھ رہے ہیں…"), status="pending"),
                resume_chunk_index=current_index + 1,
                source_type=retrieval.source_type,
                sources=retrieval.sources,
                language=self.response_language(lesson),
                answer_language=answer_language if self.language_detection_enabled else None,
                reteach=True,
            )

        retrieval = self.retriever.retrieve(question, lesson.course_id)
        tone_directive = tone_directive_for_state(lesson.latest_attention, lesson.latest_mood)
        delivered_lesson = "\n".join(
            block.text for block in lesson.delivered_blocks
            if block.type not in {"source_image", "no_face_pause"}
        )[-12000:]
        messages = [
            {"role": "system", "content": _interruption_prompt_for(answer_language)},
            {"role": "system", "content": f"Active lesson topic: {lesson.topic}\nSource-section outline: {lesson.source_outline}\nSections already taught: {lesson.covered_sections}\nCurrent section: {lesson.active_section}\nCurrent line:\n{current}\nLesson actually delivered so far:\n{delivered_lesson or '(none)'}\nReference material for this question:\n{retrieval.context or '(none)'}"},
        ]
        if tone_directive:
            messages.append({"role": "system", "content": f"Tone guidance: {tone_directive}"})
        messages.extend([
            *lesson.history[-6:],
            {"role": "user", "content": question},
        ])
        answer = clean_spoken_text(await self.llm.safe_call(messages, temperature=0.4))
        lesson.history.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])
        resume = current_index + 1
        return InterruptionResponse(
            question=WhiteboardBlock(id=str(uuid4()), type="student_question", text=question, status="complete"),
            answer=WhiteboardBlock(id=str(uuid4()), type="tutor_answer", text=answer, status="active"),
            continuation=WhiteboardBlock(id=str(uuid4()), type="continuation", text=localized(lesson.language, "Continuing the lesson…", "سبق جاری رکھ رہے ہیں…"), status="pending"),
            resume_chunk_index=resume,
            source_type=retrieval.source_type,
            sources=retrieval.sources,
            language=self.response_language(lesson),
            answer_language=answer_language if self.language_detection_enabled else None,
        )


    async def quiz(self, session_id: str, count: int) -> LessonQuizResponse:
        lesson = self.store.get(session_id)
        # The lesson text is the quiz source. It is populated at start() and as
        # sections are generated; as a last-resort guard (e.g. empty generated
        # section) fall back to the spoken history and the topic so the model is
        # never handed a blank lesson, which causes it to reply in plain text.
        lesson_text = lesson.lecture_text.strip()
        if not lesson_text:
            lesson_text = " ".join(
                item.get("content", "")
                for item in lesson.history
                if str(item.get("role")) == "assistant" and str(item.get("content", "")).strip()
            ).strip() or lesson.topic
        quiz_system = (
            "Create a quiz only from the supplied lesson. Return JSON with questions. "
            "Mix multiple_choice, true_false, and short_answer. Each item needs type, "
            "question, options (four for multiple choice, two for true/false, empty for "
            "short answer). Do not include answers."
        )
        if lesson.language in ("Urdu", "Roman Urdu"):
            quiz_system = f"{quiz_system} {language_instruction(output_language(lesson.language))}"
        payload = await self.llm.json([
            {"role": "system", "content": quiz_system},
            {"role": "user", "content": f"Create exactly {count} questions from this lesson:\n{lesson_text[:24000]}"},
        ])
        raw_questions = payload if isinstance(payload, list) else payload.get("questions", [])
        questions = []
        for item in raw_questions[:count]:
            kind = item.get("type", "short_answer")
            if kind not in {"multiple_choice", "true_false", "short_answer"}: kind = "short_answer"
            questions.append(LessonQuizQuestion(id=str(uuid4()), type=kind, question=str(item.get("question", "")), options=[str(option) for option in item.get("options", [])]))
        return LessonQuizResponse(questions=questions, language=self.response_language(lesson))

    async def grade_quiz(self, session_id: str, question: LessonQuizQuestion, answer: str) -> QuizGradeResponse:
        lesson = self.store.get(session_id)
        grade_system = (
            "Grade the student's answer semantically against the lesson and question. "
            "Be consistent: decide the verdict first, then write the explanation so it "
            "exactly matches that verdict. If the student's answer is right even when "
            "phrased differently, result must be correct. Return JSON with result exactly "
            "correct, partially_correct, or incorrect, plus a short explanation. Never "
            "label an answer incorrect while your explanation describes it as correct."
        )
        if lesson.language in ("Urdu", "Roman Urdu"):
            grade_system = f"{grade_system} {language_instruction(output_language(lesson.language))}"
        payload = await self.llm.json([
            {"role": "system", "content": grade_system},
            {"role": "user", "content": f"Lesson:\n{lesson.lecture_text[:16000]}\nQuestion: {question.question}\nStudent answer: {answer}"},
        ], temperature=0)
        result = payload.get("result", "incorrect")
        if result not in {"correct", "partially_correct", "incorrect"}: result = "incorrect"
        explanation = clean_markdown(str(payload.get("explanation", "")))
        reconciled = _reconcile_grade_result(result, explanation)
        if reconciled is not None:
            result = reconciled
        return QuizGradeResponse(
            result=result,
            explanation=explanation,
            language=self.response_language(lesson),
        )
