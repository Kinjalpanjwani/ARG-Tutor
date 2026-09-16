from dataclasses import dataclass, field
from threading import RLock
from uuid import uuid4
import re
import logging
from time import perf_counter

from app.core.constants import ACADEMIC_ONLY_MESSAGE
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
from app.teaching.chunks import clean_markdown, lecture_blocks
from app.teaching.router import route_request


LECTURE_PERSONA = """
You are a warm, clear, patient teacher speaking aloud in a classroom. Explain rather
than merely summarize. Preserve important source details, explain terminology, connect
ideas, and use examples, equations, or visual references when useful. Match the requested
depth. Never invent facts attributed to the source. If supplementing source material,
clearly phrase it as added background. Return valid JSON only with lesson_title,
sections (each with heading and blocks), and 2-4 follow_up_questions. Each block has
type (text, note, example, equation, graph, diagram, table, step_flow, comparison,
or number_line), content, and optional latex. For mathematics, prefer precise equation,
step_flow, number_line, graph, or diagram blocks. Do not request a source screenshot
unless the student asks to inspect the original source. Keep blocks whiteboard-sized.
""" + LANGUAGE_RULE

logger = logging.getLogger(__name__)

INTERRUPTION_PROMPT = """
The student interrupted during a lecture. Answer directly and briefly in 2-4 short,
spoken sentences, like a teacher pausing to clarify. Use the course context as the
primary source when present. Do not restart or regenerate the lecture. Do not use
Markdown.
""" + LANGUAGE_RULE


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
    depth: str = "standard"
    response_type: str = "lesson"


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


class NotebookLessonService:
    """Web port of generate_lecture → split_into_chunks → indexed teach loop."""

    def __init__(self, llm: GroqService, guard: AcademicGuard, retriever: Retriever) -> None:
        self.llm = llm
        self.guard = guard
        self.retriever = retriever
        self.store = NotebookLessonStore()

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
    def _table_fast_path(topic: str) -> tuple[str, list[WhiteboardBlock]] | None:
        match = re.fullmatch(r"(?:the )?table of (\d+)\??", topic.strip(), re.I)
        if not match: return None
        number = int(match.group(1))
        blocks = [WhiteboardBlock(id=str(uuid4()), type="heading", text=f"Table of {number}")]
        blocks.extend(WhiteboardBlock(id=str(uuid4()), type="table", text=f"{number} × {i} = {number * i}") for i in range(1, 11))
        blocks.append(WhiteboardBlock(id=str(uuid4()), type="note", text="Want to practice it?"))
        return "multiplication_table", blocks

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
                if block_type not in {"text", "note", "example", "equation", "graph", "diagram", "table", "step_flow", "comparison", "number_line"}:
                    block_type = "text"
                if content:
                    blocks.append(WhiteboardBlock(
                        id=str(uuid4()), type=block_type, text=content,
                        latex=str(item.get("latex")) if item.get("latex") else None,
                    ))
        questions = [clean_markdown(str(q)) for q in payload.get("follow_up_questions", []) if str(q).strip()][:4]
        return blocks, questions

    def _source_sections(self, topic: str, retrieval) -> list[tuple[str, str]]:
        if not retrieval.records:
            return [("Requested topic", "")]
        document_id = retrieval.records[0].document_id
        records = [record for record in self.retriever.store.records if record.document_id == document_id]
        records.sort(key=lambda record: (record.page, record.chunk_id))
        lowered = topic.casefold()
        if "every page" in lowered or "page by page" in lowered:
            requested_chapter = re.search(r"chapter\s+([\w-]+)", topic, re.I)
            if requested_chapter:
                chapter_token = requested_chapter.group(1)
                start_index = next((index for index, record in enumerate(records) if re.search(rf"\bchapter\s+{re.escape(chapter_token)}\b", record.page_content, re.I)), None)
                if start_index is not None:
                    end_index = next((index for index in range(start_index + 1, len(records)) if re.search(r"\bchapter\s+[\w-]+", records[index].page_content, re.I)), len(records))
                    records = records[start_index:end_index]
            pages: dict[int, list[str]] = {}
            for record in records:
                pages.setdefault(record.page, []).append(record.page_content)
            return [(f"Page {page}", "\n".join(text)) for page, text in pages.items()]
        if "every chapter" in lowered or "chapter" in lowered and "detail" in lowered:
            full_text = "\n\n".join(record.page_content for record in records)
            matches = list(re.finditer(r"(?im)^\s*chapter\s+([\w-]+)(?:\s*[:.-]\s*([^\n]+))?\s*$", full_text))
            if matches:
                return [
                    (
                        f"Chapter {match.group(1)}" + (f": {match.group(2).strip()}" if match.group(2) else ""),
                        full_text[match.start(): matches[index + 1].start() if index + 1 < len(matches) else len(full_text)].strip(),
                    )
                    for index, match in enumerate(matches)
                ]
            # Evidence-based fallback: short title-like first lines start semantic sections.
            inferred = []
            for record in records:
                first = next((line.strip() for line in record.page_content.splitlines() if line.strip()), "")
                words = first.split()
                if 1 < len(words) <= 10 and not first.endswith((".", "?", "!")):
                    inferred.append((first, record.page, record.page_content))
            if len(inferred) >= 2:
                return [(f"{title} (page {page})", text) for title, page, text in inferred]
        return [("Requested material", "\n\n".join(record.page_content for record in records))]

    async def _generate_section(self, topic: str, language: str, depth: str, title: str, context: str) -> tuple[list[WhiteboardBlock], list[str]]:
        depth_rules = {
            "brief": "Use 3-5 concise teaching blocks.",
            "standard": "Use 7-12 teaching blocks with an example.",
            "detailed": "Use 12-20 substantial teaching blocks with important details and examples.",
            "exhaustive": "Explain this source section thoroughly in 15-25 teaching blocks, preserving sequence and details.",
        }
        messages = [
            {"role": "system", "content": LECTURE_PERSONA},
            {"role": "system", "content": f"Follow the student's explicit request first; do not add generic importance/history unless asked. Depth: {depth}. {depth_rules[depth]}\nCurrent source section: {title}\n{context or '(No relevant attachment; use general academic knowledge.)'}"},
            {"role": "user", "content": f"Student request: {topic}\nTeach section: {title}\nLanguage: {language}"},
        ]
        try:
            return self._structured_blocks(await self.llm.json(messages, temperature=0.45))
        except Exception:
            fallback = clean_markdown(await self.llm.text(messages + [{"role": "system", "content": "JSON failed. Return a plain spoken lesson now."}], temperature=0.45))
            return lecture_blocks(fallback), []

    async def start(self, course_id: str, topic: str, language: str) -> LectureStartResponse:
        request_start = perf_counter()
        logger.info("[TUTOR] request received topic=%r", topic)
        intent = self._intent(topic)
        fast = self._table_fast_path(topic)
        if fast:
            response_type, chunks = fast
            session_id = str(uuid4())
            lecture = " ".join(block.text for block in chunks)
            self.store.put(NotebookLesson(session_id, course_id, topic, language, lecture, chunks, [], history=[{"role":"assistant","content":lecture}], follow_up_questions=[f"Can you continue the table of {topic.split()[-1].rstrip('?')} beyond 10?"], response_type=response_type))
            logger.info("[WHITEBOARD] blocks received=%d [PERF] generation: 0ms", len(chunks))
            return LectureStartResponse(session_id=session_id, topic=topic, chunks=chunks, source_type="general_knowledge", sources=[], intent=intent, depth="brief", follow_up_questions=self.store.get(session_id).follow_up_questions, response_type=response_type)
        decision = await self.guard.classify(topic)
        if not decision.allowed:
            raise ValueError(ACADEMIC_ONLY_MESSAGE)
        logger.info("[INTENT] %s [PERF] intent: %.0fms", intent, (perf_counter() - request_start) * 1000)
        retrieval_start = perf_counter()
        retrieval = self.retriever.retrieve(topic, course_id)
        logger.info("[PERF] retrieval: %.0fms", (perf_counter() - retrieval_start) * 1000)
        route = route_request(topic, retrieval)
        depth = self._depth(topic)
        chunks: list[WhiteboardBlock] = []
        followups: list[str] = []
        source_sections = self._source_sections(topic, retrieval)
        title, context = source_sections[0]
        generation_start = perf_counter()
        logger.info("[LLM] request start section=%r", title)
        section_blocks, section_questions = await self._generate_section(topic, language, depth, title, context[:24000])
        logger.info("[LLM] request success [PERF] generation: %.0fms", (perf_counter() - generation_start) * 1000)
        if len(source_sections) > 1: chunks.append(WhiteboardBlock(id=str(uuid4()), type="heading", text=title))
        chunks.extend(section_blocks); followups.extend(section_questions)
        logger.info("[WHITEBOARD] blocks received=%d", len(chunks))
        lecture = " ".join(block.text for block in chunks)
        visual_sources = []
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
        chunks = visual_sources[:2] + chunks
        if not chunks:
            raise RuntimeError("The tutor returned no teachable lecture chunks")
        session_id = str(uuid4())
        lesson = NotebookLesson(
            session_id=session_id, course_id=course_id, topic=topic,
            language=language, lecture_text=lecture, chunks=chunks,
            sources=retrieval.sources,
            history=[{"role": "assistant", "content": lecture}],
            follow_up_questions=followups[:4],
            pending_sections=source_sections[1:], depth=depth, response_type="lesson",
        )
        self.store.put(lesson)
        return LectureStartResponse(
            session_id=session_id, topic=topic, chunks=chunks,
            source_type=retrieval.source_type, sources=retrieval.sources,
            intent=intent,
            depth=depth,
            follow_up_questions=followups[:4],
            has_more_sections=len(source_sections) > 1,
            response_type="lesson",
        )

    async def next_section(self, session_id: str) -> LessonSectionResponse:
        lesson = self.store.get(session_id)
        if not lesson.pending_sections:
            return LessonSectionResponse(chunks=[], follow_up_questions=lesson.follow_up_questions[:4], has_more_sections=False)
        title, context = lesson.pending_sections.pop(0)
        logger.info("[LLM] request start section=%r", title)
        started = perf_counter()
        blocks, questions = await self._generate_section(lesson.topic, lesson.language, lesson.depth, title, context[:24000])
        blocks.insert(0, WhiteboardBlock(id=str(uuid4()), type="heading", text=title))
        lesson.chunks.extend(blocks); lesson.lecture_text += " " + " ".join(block.text for block in blocks)
        lesson.follow_up_questions = (lesson.follow_up_questions + questions)[:4]
        logger.info("[LLM] request success [PERF] generation: %.0fms [WHITEBOARD] blocks received=%d", (perf_counter() - started) * 1000, len(blocks))
        return LessonSectionResponse(chunks=blocks, follow_up_questions=lesson.follow_up_questions, has_more_sections=bool(lesson.pending_sections))

    async def interrupt(self, session_id: str, question: str, current_index: int) -> InterruptionResponse:
        lesson = self.store.get(session_id)
        retrieval = self.retriever.retrieve(question, lesson.course_id)
        current = lesson.chunks[min(current_index, len(lesson.chunks) - 1)].text
        answer = clean_markdown(await self.llm.text([
            {"role": "system", "content": INTERRUPTION_PROMPT},
            {"role": "system", "content": f"Original lecture:\n{lesson.lecture_text}\nCurrent line:\n{current}\nReference material:\n{retrieval.context or '(none)'}"},
            *lesson.history[-6:],
            {"role": "user", "content": question},
        ], temperature=0.4))
        lesson.history.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])
        resume = min(current_index + 1, len(lesson.chunks))
        return InterruptionResponse(
            question=WhiteboardBlock(id=str(uuid4()), type="student_question", text=question, status="complete"),
            answer=WhiteboardBlock(id=str(uuid4()), type="tutor_answer", text=answer, status="active"),
            continuation=WhiteboardBlock(id=str(uuid4()), type="continuation", text="Continuing the lesson…", status="pending"),
            resume_chunk_index=resume,
            source_type=retrieval.source_type,
            sources=retrieval.sources,
        )

    async def quiz(self, session_id: str, count: int) -> LessonQuizResponse:
        lesson = self.store.get(session_id)
        payload = await self.llm.json([
            {"role": "system", "content": "Create a quiz only from the supplied lesson. Return JSON with questions. Mix multiple_choice, true_false, and short_answer. Each item needs type, question, options (four for multiple choice, two for true/false, empty for short answer). Do not include answers."},
            {"role": "user", "content": f"Create exactly {count} questions from this lesson:\n{lesson.lecture_text[:24000]}"},
        ])
        raw_questions = payload if isinstance(payload, list) else payload.get("questions", [])
        questions = []
        for item in raw_questions[:count]:
            kind = item.get("type", "short_answer")
            if kind not in {"multiple_choice", "true_false", "short_answer"}: kind = "short_answer"
            questions.append(LessonQuizQuestion(id=str(uuid4()), type=kind, question=str(item.get("question", "")), options=[str(option) for option in item.get("options", [])]))
        return LessonQuizResponse(questions=questions)

    async def grade_quiz(self, session_id: str, question: LessonQuizQuestion, answer: str) -> QuizGradeResponse:
        lesson = self.store.get(session_id)
        payload = await self.llm.json([
            {"role": "system", "content": "Grade the student's answer semantically using the lesson. Return JSON with result exactly correct, partially_correct, or incorrect, plus a short explanation."},
            {"role": "user", "content": f"Lesson:\n{lesson.lecture_text[:16000]}\nQuestion: {question.question}\nStudent answer: {answer}"},
        ])
        result = payload.get("result", "incorrect")
        if result not in {"correct", "partially_correct", "incorrect"}: result = "incorrect"
        return QuizGradeResponse(result=result, explanation=clean_markdown(str(payload.get("explanation", ""))))
