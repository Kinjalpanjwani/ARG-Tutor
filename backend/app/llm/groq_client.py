import json
import re
from typing import Any

from groq import AsyncGroq


class GroqConfigurationError(RuntimeError):
    pass


class GroqServiceError(RuntimeError):
    pass


class GroqService:
    def __init__(self, api_key: str | None, model: str, whisper_model: str) -> None:
        self.model = model
        self.whisper_model = whisper_model
        self._client = AsyncGroq(api_key=api_key) if api_key else None

    @property
    def client(self) -> AsyncGroq:
        if self._client is None:
            raise GroqConfigurationError("GROQ_API_KEY is not configured")
        return self._client

    async def text(self, messages: list[dict[str, str]], temperature: float = 0.4) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=temperature
            )
            return (response.choices[0].message.content or "").strip()
        except GroqConfigurationError:
            raise
        except Exception as exc:
            raise GroqServiceError(f"Groq request failed: {exc}") from exc

    async def json(self, messages: list[dict[str, str]], temperature: float = 0.2) -> Any:
        raw = await self.text(messages, temperature)
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I | re.S)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise GroqServiceError("Groq returned invalid structured JSON") from exc

    async def transcribe(self, path: str) -> str:
        try:
            with open(path, "rb") as audio:
                result = await self.client.audio.transcriptions.create(
                    model=self.whisper_model, file=audio
                )
            return result.text.strip()
        except GroqConfigurationError:
            raise
        except Exception as exc:
            raise GroqServiceError(f"Transcription failed: {exc}") from exc
