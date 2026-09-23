import asyncio
import json
import re
from typing import Any

from groq import APIConnectionError, AsyncGroq


class GroqConfigurationError(RuntimeError):
    pass


class GroqServiceError(RuntimeError):
    pass


# Backoff (seconds) applied between retries for rate limits / transient failures.
_RETRY_BACKOFF_SECONDS = (0.5, 1.5, 3.0, 6.0)
# Per-message content budget (characters) applied when a request must be tightened
# after a token-limit / "request too long" error. Keeps the instruction head and the
# final question tail, discarding the middle of oversized context.
_CONTENT_BUDGET = 12000


def _status_code(exc: Exception) -> int | None:
    return getattr(exc, "status_code", None)


def _error_class(exc: Exception) -> str:
    """Bucket Groq failures so the client can react appropriately.

    Returns one of: auth, rate_limit, transient, token_limit, other.
    """
    status = _status_code(exc)
    message = str(exc)
    lowered = message.lower()
    if status in (401, 403):
        return "auth"
    if status == 429 or any(token in lowered for token in (
        "rate_limit", "rate limit", "too many requests", " tpm", " rpm", "exceeded your current quota",
    )):
        return "rate_limit"
    if status in (408, 409, 500, 502, 503, 504, 529) or any(token in lowered for token in (
        "overloaded", "temporarily", "try again", "internal server error", "timed out", "timeout",
        "connection error", "server error",
    )) or isinstance(exc, APIConnectionError):
        return "transient"
    if status in (400, 413) or any(token in lowered for token in (
        "context_length_exceeded", "context length", "maximum context", "request too long",
        "request too large", "too many tokens", "token limit", "exceeds the model",
        "prompt is too long", "input is too long", "max tokens", "history length exceeded",
    )):
        return "token_limit"
    return "other"


def _keep_first_and_last(messages: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    """Keep the first system message (assumed at index 0) and the last (limit-1) messages."""
    if len(messages) <= limit:
        return messages
    return [messages[0]] + messages[-(limit - 1):]


def _cap_content(content: str, budget: int = _CONTENT_BUDGET) -> str:
    """Truncate an oversized message from the middle, keeping head + tail."""
    if len(content) <= budget:
        return content
    head_budget = int(budget * 0.7)
    tail_budget = budget - head_budget
    return f"{content[:head_budget]}\n…[truncated]…\n{content[-tail_budget:]}"


def _tighten(messages: list[dict[str, str]], limit: int, budget: int = _CONTENT_BUDGET) -> list[dict[str, str]]:
    """Stronger window for retry: fewer messages and each message content-capped."""
    trimmed = _keep_first_and_last(messages, limit)
    return [
        {**message, "content": _cap_content(message["content"], budget)}
        if isinstance(message.get("content"), str)
        else message
        for message in trimmed
    ]


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

    async def _complete(self, messages: list[dict[str, str]], *, temperature: float, model: str) -> str:
        response = await self.client.chat.completions.create(
            model=model, messages=messages, temperature=temperature
        )
        return (response.choices[0].message.content or "").strip()

    async def _chat_with_retries(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        model: str,
        trim_history: int = 8,
        retry_trim: int = 4,
    ) -> str:
        """Run a completion with backoff retries and progressive input tightening.

        - rate_limit / transient: retry with backoff; after the first retry the window
          also shrinks to retry_trim with per-message caps.
        - token_limit ("request too long", context length, ...): immediately tighten the
          window (retry_trim + content caps) and retry.
        - auth / other: raise immediately.
        """
        working = _keep_first_and_last(messages, trim_history)
        last_error: Exception | None = None
        last_class = "other"
        max_attempts = 1 + len(_RETRY_BACKOFF_SECONDS)
        for attempt in range(max_attempts):
            try:
                return await self._complete(working, temperature=temperature, model=model)
            except GroqServiceError:
                raise
            except Exception as exc:
                last_error = exc
                last_class = _error_class(exc)
                if last_class == "auth":
                    raise GroqServiceError(f"Groq request failed: {exc}") from exc
                if attempt + 1 >= max_attempts:
                    break
                if last_class == "token_limit" or attempt >= 1:
                    working = _tighten(messages, retry_trim)
                await asyncio.sleep(_RETRY_BACKOFF_SECONDS[attempt])
        if last_class in {"rate_limit", "transient", "token_limit"}:
            # Lesson continuity fallback: keep the spoken flow alive until the
            # service frees up. JSON callers surface the parse error instead.
            return "Give me just a second, I'm gathering my thoughts..."
        raise GroqServiceError(f"Groq request failed: {last_error}") from last_error

    async def text(self, messages: list[dict[str, str]], temperature: float = 0.4) -> str:
        try:
            return await self._chat_with_retries(messages, temperature=temperature, model=self.model)
        except GroqConfigurationError:
            raise

    async def safe_call(self, messages: list[dict[str, str]], *, temperature: float = 0.4, model: str | None = None, trim_history: int = 8, retry_trim: int = 4) -> str:
        """Call Groq with automatic history trimming, backoff retries, and escalation
        on rate limits / token-limit errors.
        Parameters:
            messages: list of message dicts.
            temperature: sampling temperature.
            model: optional model override; defaults to self.model.
            trim_history: number of recent messages to keep (including the first system message).
            retry_trim: number of messages to keep on retry (should be smaller).
        Returns:
            The raw response text.
        """
        chosen_model = model or self.model
        return await self._chat_with_retries(
            messages, temperature=temperature, model=chosen_model,
            trim_history=trim_history, retry_trim=retry_trim,
        )

    async def json(self, messages: list[dict[str, str]], temperature: float = 0.4) -> dict:
        """Call Groq expecting JSON output and return parsed dict.
        Uses safe_call for trimming/retry and then parses the response.
        """
        raw = await self.safe_call(messages, temperature=temperature)
        # Clean potential markdown code fences
        cleaned = re.sub(r"^```json\n?|\n```$", "", raw.strip(), flags=re.MULTILINE)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise GroqServiceError(f"Failed to parse JSON response: {exc}\nRaw: {raw}")

    async def transcribe(self, file_path: str, language: str | None = None) -> str:
        """Transcribe an audio file using Groq Whisper endpoint.

        language: optional ISO-639-1 code (e.g. "ur"). When omitted, Whisper
        auto-detects the spoken language, which can mislabel Urdu/Hindustani
        speech as Hindi ("hi").
        """
        try:
            with open(file_path, "rb") as f:
                kwargs: dict[str, Any] = {"model": self.whisper_model}
                if language:
                    kwargs["language"] = language
                transcription = await self.client.audio.transcriptions.create(
                    file=(re.split(r"[\\/]", file_path)[-1], f.read()),
                    **kwargs,
                )
            return (transcription.text or "").strip()
        except Exception as exc:
            raise GroqServiceError(f"Groq transcription failed: {exc}") from exc