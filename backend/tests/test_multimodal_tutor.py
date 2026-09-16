from pathlib import Path
from zipfile import ZipFile

from PIL import Image, ImageDraw

from app.rag.chunker import DocumentChunker
from app.rag.ingestion import IngestionService
from app.rag.loader import DocumentLoader
from app.rag.retriever import RetrievalResult, Retriever
from app.rag.vector_store import VectorStore
from app.rag.vector_store import ChunkRecord
from app.storage import DocumentRepository
from app.teaching.router import route_request
from app.teaching.notebook_lesson import NotebookLessonService
from tests.fakes import FakeEmbeddings
from tests.fakes import FakeGuard
import pytest


def _text_image(path: Path, text: str) -> None:
    image = Image.new("RGB", (1200, 350), "white")
    ImageDraw.Draw(image).text((50, 120), text, fill="black", font_size=42)
    image.save(path)


def test_image_and_scanned_pdf_use_ocr_and_retain_visual(tmp_path):
    image_path = tmp_path / "equation.png"
    _text_image(image_path, "x^2 + 5x + 6 = 0")
    pdf_path = tmp_path / "scan.pdf"
    source = Image.open(image_path)
    pages = [source.copy(), source.copy(), source.copy()]
    pages[0].save(pdf_path, "PDF", save_all=True, append_images=pages[1:])
    loader = DocumentLoader(tmp_path / "visuals")

    image_page = loader.load(image_path, "image-doc")[0]
    scan_page = loader.load(pdf_path, "scan-doc")[0]

    assert "5x" in image_page.text
    assert "5x" in scan_page.text
    assert image_page.visual_paths[0].exists()
    assert scan_page.visual_paths[0].exists()


def test_pptx_extracts_slide_text_and_number(tmp_path):
    path = tmp_path / "lesson.pptx"
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "ppt/slides/slide3.xml",
            '<p:sld xmlns:p="p" xmlns:a="a"><a:t>Gradient descent diagram</a:t><a:t>Learning rate</a:t></p:sld>',
        )
    page = DocumentLoader(tmp_path / "visuals").load(path, "slides")[0]
    assert page.page == 3
    assert "Gradient descent diagram" in page.text


def test_router_uses_general_knowledge_without_relevant_material():
    retrieval = RetrievalResult(context="", sources=[], records=[])
    assert route_request("Explain prime numbers", retrieval).intent == "general_topic"
    assert route_request("x^2 + 5x + 6 = 0", retrieval).intent == "solve_equation"


def test_removal_rebuilds_index_without_document(tmp_path):
    embeddings = FakeEmbeddings()
    store = VectorStore(tmp_path / "vectors", dimension=4)
    documents = DocumentRepository(tmp_path / "documents.json")
    retriever = Retriever(embeddings, store, top_k=10, threshold=-1)
    ingestion = IngestionService(
        DocumentLoader(tmp_path / "visuals"), DocumentChunker(100, 10),
        embeddings, store, documents, retriever,
    )
    first_path = tmp_path / "A.txt"; first_path.write_text("alpha astronomy stars", encoding="utf-8")
    second_path = tmp_path / "B.txt"; second_path.write_text("beta biology cells", encoding="utf-8")
    first, _ = ingestion.ingest(first_path, first_path.read_bytes(), "course", "notes")
    second, _ = ingestion.ingest(second_path, second_path.read_bytes(), "course", "notes")

    _, removed = ingestion.remove(second.document_id, "course")

    assert removed == second.chunks_indexed
    assert all(item.document_id != second.document_id for item in store.records)
    assert any(item.document_id == first.document_id for item in store.records)


def test_chapter_and_page_sections_preserve_order(tmp_path):
    records = [
        ChunkRecord(page_content="Chapter 1: Start\nFirst details\nChapter 2: Middle", document_id="book", course_id="c", document_name="book.md", document_type="book", page=1, chunk_id="1"),
        ChunkRecord(page_content="Second details\nChapter 3: End\nThird details", document_id="book", course_id="c", document_name="book.md", document_type="book", page=2, chunk_id="2"),
    ]
    service = NotebookLessonService(None, None, type("R", (), {"store": type("S", (), {"records": records})()})())
    retrieval = type("Result", (), {"records": records})()
    chapters = service._source_sections("Explain every chapter in detail", retrieval)
    pages = service._source_sections("Explain every page", retrieval)
    assert [title for title, _ in chapters] == ["Chapter 1: Start", "Chapter 2: Middle", "Chapter 3: End"]
    assert [title for title, _ in pages] == ["Page 1", "Page 2"]


@pytest.mark.asyncio
async def test_table_fast_path_is_exact_and_skips_llm():
    class FailingLLM:
        async def json(self, *args, **kwargs):
            raise AssertionError("simple table must not call Groq")
    retriever = type("R", (), {"store": type("S", (), {"records": []})()})()
    service = NotebookLessonService(FailingLLM(), FakeGuard(), retriever)
    result = await service.start("course", "table of 2?", "English")
    assert result.response_type == "multiplication_table"
    assert [block.text for block in result.chunks[1:11]] == [f"2 × {i} = {2*i}" for i in range(1, 11)]


def test_title_like_pages_support_headingless_book_sections():
    records = [
        ChunkRecord(page_content="The Journey Begins\nMira left home.", document_id="book", course_id="c", document_name="book.md", document_type="book", page=2, chunk_id="1"),
        ChunkRecord(page_content="The Forest\nShe crossed the trees.", document_id="book", course_id="c", document_name="book.md", document_type="book", page=3, chunk_id="2"),
    ]
    service = NotebookLessonService(None, None, type("R", (), {"store": type("S", (), {"records": records})()})())
    retrieval = type("Result", (), {"records": records})()
    sections = service._source_sections("Explain every chapter in detail", retrieval)
    assert [title for title, _ in sections] == ["The Journey Begins (page 2)", "The Forest (page 3)"]
