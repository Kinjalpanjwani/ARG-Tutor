import json
from pathlib import Path
from threading import RLock
from typing import Generic, TypeVar

from pydantic import BaseModel

from app.schemas.courses import Course, CourseCreate
from app.schemas.documents import DocumentRecord

T = TypeVar("T", bound=BaseModel)


class JsonRepository(Generic[T]):
    def __init__(self, path: Path, model: type[T]) -> None:
        self.path = path
        self.model = model
        self._lock = RLock()
        self._items: dict[str, T] = {}
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            self._items = {key: model.model_validate(value) for key, value in raw.items()}

    def all(self) -> list[T]:
        with self._lock:
            return list(self._items.values())

    def get(self, key: str) -> T | None:
        with self._lock:
            return self._items.get(key)

    def put(self, key: str, item: T) -> T:
        with self._lock:
            self._items[key] = item
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {k: v.model_dump(mode="json") for k, v in self._items.items()}
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(self.path)
            return item

    def delete(self, key: str) -> T | None:
        with self._lock:
            item = self._items.pop(key, None)
            if item is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                payload = {k: v.model_dump(mode="json") for k, v in self._items.items()}
                temporary = self.path.with_suffix(".tmp")
                temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                temporary.replace(self.path)
            return item


class CourseRepository(JsonRepository[Course]):
    def __init__(self, path: Path) -> None:
        super().__init__(path, Course)

    def create(self, data: CourseCreate) -> Course:
        course = Course(**data.model_dump())
        return self.put(course.course_id, course)


class DocumentRepository(JsonRepository[DocumentRecord]):
    def __init__(self, path: Path) -> None:
        super().__init__(path, DocumentRecord)

    def find_duplicate(self, course_id: str, sha256: str) -> DocumentRecord | None:
        return next(
            (doc for doc in self.all() if doc.course_id == course_id and doc.sha256 == sha256),
            None,
        )
