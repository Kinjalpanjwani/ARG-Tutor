from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_container
from app.container import Container
from app.schemas.courses import Course, CourseCreate

router = APIRouter(prefix="/courses", tags=["courses"])


@router.post("", response_model=Course, status_code=status.HTTP_201_CREATED)
async def create_course(data: CourseCreate, container: Container = Depends(get_container)) -> Course:
    return container.courses.create(data)


@router.get("", response_model=list[Course])
async def list_courses(container: Container = Depends(get_container)) -> list[Course]:
    return container.courses.all()


@router.get("/{course_id}", response_model=Course)
async def get_course(course_id: str, container: Container = Depends(get_container)) -> Course:
    course = container.courses.get(course_id)
    if not course:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Course not found")
    return course
