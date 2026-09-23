from app.teaching.chunks import (
    BLOCK_SPLIT_LIMIT,
    clean_markdown,
    clean_spoken_text,
    lecture_blocks,
    split_into_chunks,
    split_prose_for_whiteboard,
)


def test_notebook_splitter_keeps_urdu_and_english_sentence_boundaries() -> None:
    chunks = split_into_chunks("First idea. دوسری بات۔ Is this clear? جی ہاں؟")
    assert chunks == ["First idea.", "دوسری بات۔", "Is this clear?", "جی ہاں؟"]


def test_markdown_is_removed_and_blocks_are_structured() -> None:
    text = "**Definition:** A polynomial uses variables. Example: 3x + 5. Remember the exponent is whole."
    assert "**" not in clean_markdown(text)
    blocks = lecture_blocks(text)
    assert [block.type for block in blocks] == ["heading", "example", "note"]
    assert all(block.status == "pending" for block in blocks)


def test_clean_spoken_text_re_separates_numbered_bullets() -> None:
    text = "Here are the key points. **1.** First idea. 2. Second idea. 3. Third idea."
    cleaned = clean_spoken_text(text)
    assert "**" not in cleaned
    assert "1. First idea." in cleaned.splitlines()
    assert "2. Second idea." in cleaned.splitlines()
    assert "3. Third idea." in cleaned.splitlines()


def test_clean_spoken_text_unwraps_stray_json_payload() -> None:
    text = '{"lesson_title": "Polynomials", "sections": [{"heading": "Intro", "blocks": [{"type": "text", "content": "A polynomial is an integer-power expression."}]}]}'
    cleaned = clean_spoken_text(text)
    assert "{" not in cleaned
    assert "Poly" in cleaned
    assert "integer-power" in cleaned
    assert "lesson_title" not in cleaned
    assert "sections" not in cleaned


def test_clean_spoken_text_leaves_normal_prose_unchanged() -> None:
    text = "A polynomial sums variables raised to whole-number powers."
    assert clean_spoken_text(text) == text


def test_split_prose_for_whiteboard_splits_oversized_prose() -> None:
    text = "First sentence about polynomials is short. " + "B" * 400 + ". Trailing sentence is here too."
    pieces = split_prose_for_whiteboard(text, limit=120)
    assert len(pieces) > 1
    assert all(len(piece) <= 120 for piece in pieces)
    assert pieces[0] == "First sentence about polynomials is short."


def test_split_prose_for_whiteboard_keeps_small_text_whole() -> None:
    text = "Short paragraph."
    assert split_prose_for_whiteboard(text) == ["Short paragraph."]
    assert split_prose_for_whiteboard("") == []


import pytest
from app.teaching.notebook_lesson import NotebookLessonService, NotebookLesson
from app.schemas.lesson import WhiteboardBlock
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore
from app.rag.vector_store import ChunkRecord
from app.rag.retriever import RetrievalResult
from app.schemas.common import Source
from tests.fakes import FakeEmbeddings, FakeGuard


class FakeLLMWithSections:
    async def json(self, messages, temperature=0.45):
        return {
            "sections": [
                {
                    "heading": "Pending Section Generated",
                    "blocks": [
                        {"type": "text", "content": "First generated body block"},
                        {"type": "text", "content": "Second generated body block"},
                    ],
                }
            ],
            "follow_up_questions": ["Question 1?"],
        }

    async def text(self, messages, temperature=0.6):
        return "Plain text fallback"


@pytest.mark.asyncio
async def test_next_section_serves_chunks_before_pending_sections(tmp_path):
    retriever = Retriever(FakeEmbeddings(), VectorStore(tmp_path, 4), threshold=-1)
    service = NotebookLessonService(FakeLLMWithSections(), FakeGuard(), retriever)

    session_id = "test-session-19-chunks"
    initial_stored_blocks = [WhiteboardBlock(id=f"b_{i}", type="text", text=f"Block {i}") for i in range(1, 20)]
    lesson = NotebookLesson(
        session_id=session_id,
        course_id="c",
        topic="Math",
        language="English",
        lecture_text="Initial lecture",
        chunks=list(initial_stored_blocks),  # 19 blocks
        sources=[],
        pending_sections=[("Pending Section 1", "Section context...")],
    )
    service.store.put(lesson)

    # First 18 calls: pops 1 block each time, has_more_sections remains True, pending_sections untouched
    for i in range(1, 19):
        resp = await service.next_section(session_id)
        assert len(resp.chunks) == 1
        assert resp.chunks[0].id == f"b_{i}"
        assert resp.has_more_sections is True
        # Ensure pending_sections is untouched
        assert len(lesson.pending_sections) == 1
        assert len(lesson.chunks) == 19 - i

    # 19th call: pops the last remaining block from lesson.chunks
    resp_19 = await service.next_section(session_id)
    assert len(resp_19.chunks) == 1
    assert resp_19.chunks[0].id == "b_19"
    # Still True because pending_sections has 1 section!
    assert resp_19.has_more_sections is True
    assert len(lesson.chunks) == 0
    assert len(lesson.pending_sections) == 1

    # 20th call: lesson.chunks is now empty, so it pops pending_sections and generates new section!
    resp_20 = await service.next_section(session_id)
    assert len(lesson.pending_sections) == 0
    # Returns 1st block of new section (heading: 'Pending Section Generated' inserted as heading)
    assert len(resp_20.chunks) == 1
    assert resp_20.has_more_sections is True
    # The remaining blocks of the new section are now stored in lesson.chunks
    assert len(lesson.chunks) > 0


@pytest.mark.asyncio
async def test_next_section_serves_chunks_with_no_pending_sections(tmp_path):
    retriever = Retriever(FakeEmbeddings(), VectorStore(tmp_path, 4), threshold=-1)
    service = NotebookLessonService(FakeLLMWithSections(), FakeGuard(), retriever)

    session_id = "test-session-no-pending"
    # 3 blocks in chunks, 0 pending sections
    initial_stored_blocks = [WhiteboardBlock(id=f"b_{i}", type="text", text=f"Block {i}") for i in range(1, 4)]
    lesson = NotebookLesson(
        session_id=session_id,
        course_id="c",
        topic="Math",
        language="English",
        lecture_text="Initial lecture",
        chunks=list(initial_stored_blocks),
        sources=[],
        pending_sections=[],
    )
    service.store.put(lesson)

    # Block 1
    r1 = await service.next_section(session_id)
    assert r1.chunks[0].id == "b_1"
    assert r1.has_more_sections is True

    # Block 2
    r2 = await service.next_section(session_id)
    assert r2.chunks[0].id == "b_2"
    assert r2.has_more_sections is True

    # Block 3 (last one) - all delivered now, pending_sections is empty, so has_more_sections must be False!
    r3 = await service.next_section(session_id)
    assert r3.chunks[0].id == "b_3"
    assert r3.has_more_sections is False

    # Block 4 (empty)
    r4 = await service.next_section(session_id)
    assert len(r4.chunks) == 0
    assert r4.has_more_sections is False


def test_normalize_block_sizes_splits_oversized_prose_blocks():
    from app.teaching.notebook_lesson import _normalize_block_sizes

    big = "W" * (BLOCK_SPLIT_LIMIT + 50)
    blocks = [
        WhiteboardBlock(id="a", type="text", text=big),
        WhiteboardBlock(id="b", type="heading", text=big),
        WhiteboardBlock(id="c", type="note", text="tiny"),
    ]
    normalized = _normalize_block_sizes(blocks)
    assert all(len(block.text) <= BLOCK_SPLIT_LIMIT for block in normalized if block.type != "heading")
    assert normalized[0].id != normalized[1].id
    assert all(block.type == "text" for block in normalized[:2])
    assert normalized[-1] == blocks[-1]


def _record(document_id: str, document_name: str, page: int, text: str) -> ChunkRecord:
    return ChunkRecord(
        page_content=text,
        document_id=document_id,
        course_id="course",
        document_name=document_name,
        document_type="course_document",
        page=page,
        chunk_id=f"{document_id}:{page}",
    )


def test_comprehensive_lesson_builds_sections_from_every_uploaded_document(tmp_path):
    retriever = Retriever(FakeEmbeddings(), VectorStore(tmp_path, 4), threshold=-1)
    records = [
        _record("doc-a", "first.pdf", 1, "Chapter 1\nOrigins and causes."),
        _record("doc-b", "second.pdf", 1, "Chapter 2\nConsequences and change."),
    ]
    retriever.store.records.extend(records)
    service = NotebookLessonService(FakeLLMWithSections(), FakeGuard(), retriever)
    retrieval = RetrievalResult(
        context=records[1].page_content,
        sources=[Source(document_id="doc-b", document_name="second.pdf", page=1, chunk_id="doc-b:1", score=1.0)],
        records=[records[1]],
    )

    sections = service._source_sections("Teach all uploaded chapters", retrieval, "course")

    assert [title.split(" — ")[0] for title, _ in sections] == ["first.pdf", "second.pdf"]
    assert "Origins and causes" in sections[0][1]
    assert "Consequences and change" in sections[1][1]


def test_lesson_state_tracks_outline_progress_explanations_and_questions():
    lesson = NotebookLesson(
        session_id="s", course_id="c", topic="Teach both chapters", language="English",
        lecture_text="", chunks=[], sources=[],
        source_outline=["Chapter one", "Chapter two"],
        covered_sections=["Chapter one"], active_section="Chapter one",
        pending_sections=[("Chapter two", "source")],
        delivered_blocks=[WhiteboardBlock(id="b", type="text", text="The first cause led to the second event.")],
        history=[
            {"role": "user", "content": "Why did that happen?"},
            {"role": "assistant", "content": "Because the pressure had accumulated."},
        ],
    )

    state = NotebookLessonService._lesson_state(lesson, "Chapter two")

    assert "Teach both chapters" in state
    assert "Chapter one" in state and "Chapter two" in state
    assert "first cause led" in state
    assert "Why did that happen?" in state

