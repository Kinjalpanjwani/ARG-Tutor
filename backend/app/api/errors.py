from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.llm.groq_client import GroqConfigurationError, GroqServiceError


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(GroqConfigurationError)
    async def missing_key(_: Request, exc: GroqConfigurationError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(GroqServiceError)
    async def provider_error(_: Request, exc: GroqServiceError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(exc)})
