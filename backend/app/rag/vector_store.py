import json
from pathlib import Path
from threading import RLock

import faiss
import numpy as np
from pydantic import BaseModel


class ChunkRecord(BaseModel):
    page_content: str
    document_id: str
    course_id: str
    document_name: str
    document_type: str
    page: int
    chunk_id: str
    subject: str | None = None
    student_level: str | None = None
    visual_url: str | None = None


class VectorStore:
    def __init__(self, directory: Path, dimension: int = 384) -> None:
        self.directory = directory
        self.index_path = directory / "index.faiss"
        self.metadata_path = directory / "chunks.json"
        self.dimension = dimension
        self._lock = RLock()
        directory.mkdir(parents=True, exist_ok=True)
        self.index = faiss.read_index(str(self.index_path)) if self.index_path.exists() else faiss.IndexFlatIP(dimension)
        if self.metadata_path.exists():
            raw = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            self.records = [ChunkRecord.model_validate(item) for item in raw]
        else:
            self.records: list[ChunkRecord] = []
        if self.index.ntotal != len(self.records):
            raise RuntimeError("FAISS index and metadata are inconsistent")

    def add(self, records: list[ChunkRecord], embeddings: np.ndarray) -> None:
        if not records:
            return
        vectors = np.asarray(embeddings, dtype="float32")
        if vectors.shape != (len(records), self.dimension):
            raise ValueError("Embedding dimensions do not match vector store")
        faiss.normalize_L2(vectors)
        with self._lock:
            self.index.add(vectors)
            self.records.extend(records)
            self._persist()

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[ChunkRecord, float]]:
        with self._lock:
            if self.index.ntotal == 0:
                return []
            vector = np.asarray([query], dtype="float32")
            faiss.normalize_L2(vector)
            scores, indexes = self.index.search(vector, min(top_k, self.index.ntotal))
            return [
                (self.records[int(index)], float(score))
                for score, index in zip(scores[0], indexes[0])
                if index >= 0
            ]

    def replace_records(self, records: list[ChunkRecord], embeddings: np.ndarray) -> None:
        vectors = np.asarray(embeddings, dtype="float32")
        with self._lock:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.records = []
            if records:
                if vectors.shape != (len(records), self.dimension):
                    raise ValueError("Embedding dimensions do not match vector store")
                faiss.normalize_L2(vectors)
                self.index.add(vectors)
                self.records = list(records)
            self._persist()

    def _persist(self) -> None:
        temp_index = self.index_path.with_suffix(".tmp.faiss")
        temp_meta = self.metadata_path.with_suffix(".tmp")
        faiss.write_index(self.index, str(temp_index))
        temp_meta.write_text(
            json.dumps([record.model_dump() for record in self.records], ensure_ascii=False),
            encoding="utf-8",
        )
        temp_index.replace(self.index_path)
        temp_meta.replace(self.metadata_path)
