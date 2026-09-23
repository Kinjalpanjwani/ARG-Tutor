from datetime import datetime, timezone

from app.schemas.lesson import (
    ArchiveBlock,
    LessonHistoryEntry,
    LessonRecord,
    QuizResultsPayload,
)
from app.storage import JsonRepository
from app.teaching.notebook_lesson import NotebookLessonStore


class LessonArchiveService:
    """Persists read-only snapshots of finished lessons keyed by session_id."""

    def __init__(self, lesson_store: NotebookLessonStore, repository: JsonRepository[LessonRecord]) -> None:
        self.lesson_store = lesson_store
        self.repository = repository

    def _snapshot(self, lesson) -> LessonRecord:
        now = datetime.now(timezone.utc)
        blocks = [ArchiveBlock(type=block.type, text=block.text, interrupt_type=block.interrupt_type) for block in lesson.chunks]
        return LessonRecord(
            session_id=lesson.session_id,
            course_id=lesson.course_id,
            topic=lesson.topic,
            language=lesson.language,
            created_at=now,
            updated_at=now,
            blocks=blocks,
            follow_up_questions=lesson.follow_up_questions,
        )

    def archive(self, session_id: str, blocks: list[ArchiveBlock]) -> LessonRecord:
        lesson = self.lesson_store.get(session_id)
        existing = self.repository.get(session_id)
        now = datetime.now(timezone.utc)
        blocks = blocks or self._snapshot(lesson).blocks
        record = LessonRecord(
            session_id=lesson.session_id,
            course_id=lesson.course_id,
            topic=lesson.topic,
            language=lesson.language,
            created_at=existing.created_at if existing else now,
            updated_at=now,
            blocks=blocks,
            follow_up_questions=lesson.follow_up_questions,
            quiz_score=existing.quiz_score if existing else None,
            quiz_total=existing.quiz_total if existing else None,
            quiz_results=existing.quiz_results if existing else [],
            quiz_at=existing.quiz_at if existing else None,
        )
        return self.repository.put(session_id, record)

    def update_quiz(self, session_id: str, payload: QuizResultsPayload) -> LessonRecord:
        lesson = self.lesson_store.get(session_id)
        now = datetime.now(timezone.utc)
        existing = self.repository.get(session_id)
        if existing is not None:
            existing.updated_at = now
            existing.quiz_score = payload.score
            existing.quiz_total = payload.total
            existing.quiz_results = payload.results
            existing.quiz_at = now
            return self.repository.put(session_id, existing)
        record = self._snapshot(lesson)
        record.updated_at = now
        record.quiz_score = payload.score
        record.quiz_total = payload.total
        record.quiz_results = payload.results
        record.quiz_at = now
        return self.repository.put(session_id, record)

    def list(self, course_id: str | None) -> list[LessonHistoryEntry]:
        records = self.repository.all()
        if course_id:
            records = [record for record in records if record.course_id == course_id]
        records.sort(key=lambda record: record.updated_at, reverse=True)
        return [
            LessonHistoryEntry(
                session_id=record.session_id,
                topic=record.topic,
                language=record.language,
                created_at=record.created_at,
                updated_at=record.updated_at,
                block_count=len(record.blocks),
                quiz_score=record.quiz_score,
                quiz_total=record.quiz_total,
            )
            for record in records
        ]

    def get(self, session_id: str) -> LessonRecord:
        record = self.repository.get(session_id)
        if record is None:
            raise KeyError(session_id)
        return record