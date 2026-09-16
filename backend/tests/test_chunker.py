from app.rag.chunker import DocumentChunker
from app.rag.loader import PageText


def test_chunking_preserves_page_and_overlap() -> None:
    chunks = DocumentChunker(chunk_size=30, chunk_overlap=5).split(
        [PageText(page=7, text="Alpha concept. Beta concept. Gamma concept. Delta concept.")]
    )
    assert len(chunks) >= 2
    assert all(chunk.page == 7 and chunk.text for chunk in chunks)
