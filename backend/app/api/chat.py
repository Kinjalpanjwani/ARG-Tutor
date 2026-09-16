from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_container
from app.container import Container
from app.core.constants import ACADEMIC_ONLY_MESSAGE
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(data: ChatRequest, container: Container = Depends(get_container)) -> ChatResponse:
    decision = await container.guard.classify(data.message)
    if not decision.allowed:
        return ChatResponse(
            content=ACADEMIC_ONLY_MESSAGE, allowed=False,
            source_type="general_knowledge", sources=[],
        )
    if data.course_id and not container.courses.get(data.course_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Course not found")
    retrieval = container.retriever.retrieve(data.message, data.course_id)
    content = await container.tutor.answer(
        data.message, data.student_level.value, data.language, retrieval.context
    )
    return ChatResponse(
        content=content, allowed=True, source_type=retrieval.source_type,
        sources=retrieval.sources,
    )
