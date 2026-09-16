from contextlib import asynccontextmanager
import shutil

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, courses, documents, health, lessons, quiz, sessions, speech, teaching
from app.api.errors import register_error_handlers
from app.container import build_container
from app.core.config import get_settings
from app.core.logging import configure_logging


def create_app() -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging()
        app.state.container = build_container(settings)
        try:
            yield
        finally:
            shutil.rmtree(app.state.container.runtime_dir, ignore_errors=True)

    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    @app.get("/media/{document_id}/{filename}", include_in_schema=False)
    async def runtime_media(document_id: str, filename: str, request: Request):
        if "/" in document_id or "/" in filename or ".." in document_id or ".." in filename:
            raise HTTPException(404)
        path = request.app.state.container.runtime_dir / "visuals" / document_id / filename
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path)
    for router in (
        health.router, courses.router, documents.router, sessions.router,
        teaching.router, lessons.router, chat.router, speech.router, quiz.router,
    ):
        app.include_router(router, prefix=settings.api_prefix)
    register_error_handlers(app)
    return app


app = create_app()
