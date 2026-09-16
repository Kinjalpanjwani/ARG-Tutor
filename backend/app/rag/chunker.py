from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.loader import PageText


@dataclass(frozen=True)
class TextChunk:
    page: int
    text: str


class DocumentChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "۔ ", " ", ""],
        )

    def split(self, pages: list[PageText]) -> list[TextChunk]:
        return [
            TextChunk(page=page.page, text=text.strip())
            for page in pages
            for text in self.splitter.split_text(page.text)
            if text.strip()
        ]
