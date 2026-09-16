from pathlib import Path
from uuid import uuid4

from app.rag.chunker import DocumentChunker
from app.rag.embeddings import EmbeddingService
from app.rag.loader import DocumentLoader
from app.rag.retriever import Retriever
from app.rag.vector_store import ChunkRecord, VectorStore
from app.schemas.documents import DocumentRecord
from app.storage import DocumentRepository
from app.utils.files import sha256_bytes


class IngestionService:
    def __init__(
        self,
        loader: DocumentLoader,
        chunker: DocumentChunker,
        embeddings: EmbeddingService,
        vector_store: VectorStore,
        documents: DocumentRepository,
        retriever: Retriever,
    ) -> None:
        self.loader = loader
        self.chunker = chunker
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.documents = documents
        self.retriever = retriever

    def ingest(
        self,
        path: Path,
        content: bytes,
        course_id: str,
        document_type: str,
        subject: str | None = None,
        student_level: str | None = None,
        document_name: str | None = None,
    ) -> tuple[DocumentRecord, bool]:
        digest = sha256_bytes(content)
        duplicate = self.documents.find_duplicate(course_id, digest)
        if duplicate:
            return duplicate, True
        document_id = str(uuid4())
        pages = self.loader.load(path, document_id)
        chunks = self.chunker.split(pages)
        page_visuals = {page.page: page.visual_paths for page in pages}
        display_name = document_name or path.name
        records = [
            ChunkRecord(
                page_content=chunk.text,
                document_id=document_id,
                course_id=course_id,
                document_name=display_name,
                document_type=document_type,
                page=chunk.page,
                chunk_id=f"{document_id}:{index}",
                subject=subject,
                student_level=student_level,
                visual_url=(
                    f"/media/{document_id}/{page_visuals[chunk.page][0].name}"
                    if page_visuals.get(chunk.page) else None
                ),
            )
            for index, chunk in enumerate(chunks)
        ]
        vectors = self.embeddings.embed_texts([record.page_content for record in records])
        self.vector_store.add(records, vectors)
        result = DocumentRecord(
            document_id=document_id,
            course_id=course_id,
            document_name=display_name,
            document_type=document_type,
            subject=subject,
            student_level=student_level,
            sha256=digest,
            chunks_indexed=len(records),
            stored_path=str(path),
            media_urls=sorted({record.visual_url for record in records if record.visual_url}),
        )
        self.documents.put(document_id, result)
        self.retriever.clear_cache()
        return result, False

    def remove(self, document_id: str, course_id: str) -> tuple[DocumentRecord, int]:
        record = self.documents.get(document_id)
        if not record or record.course_id != course_id:
            raise KeyError(document_id)
        kept = [item for item in self.vector_store.records if item.document_id != document_id]
        vectors = self.embeddings.embed_texts([item.page_content for item in kept]) if kept else []
        removed = len(self.vector_store.records) - len(kept)
        self.vector_store.replace_records(kept, vectors)
        self.documents.delete(document_id)
        if record.stored_path:
            Path(record.stored_path).unlink(missing_ok=True)
        visual_dir = self.loader.visuals_dir / document_id if self.loader.visuals_dir else None
        if visual_dir and visual_dir.exists():
            import shutil
            shutil.rmtree(visual_dir)
        self.retriever.clear_cache()
        return record, removed
