from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import get_container
from app.api.documents import router
from app.rag.retriever import RetrievalResult


class Courses:
    def get(self, course_id):
        return object() if course_id == "course-1" else None


class Retriever:
    def retrieve(self, query, course_id):
        return RetrievalResult(context="", sources=[], records=[])


class Container:
    courses = Courses()
    retriever = Retriever()


def test_retrieval_preview_hides_no_pipeline_failure() -> None:
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_container] = lambda: Container()
    response = TestClient(app).post(
        "/api/documents/retrieve-preview",
        json={"course_id": "course-1", "query": "recursion"},
    )
    assert response.status_code == 200
    assert response.json()["source_type"] == "general_knowledge"
