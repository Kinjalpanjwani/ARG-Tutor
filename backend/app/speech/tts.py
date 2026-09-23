import logging
from pathlib import Path
from uuid import uuid4
import asyncio
import random

import edge_tts

from app.language.routing import voice_for
from app.speech.urdu_loanwords import transliterate_loanwords_for_urdu

logger = logging.getLogger(__name__)


_TTS_RETRIES = 3
_TTS_BASE_BACKOFF = 0.6
_TTS_TIMEOUT = 30
_TTS_MAX_CONCURRENCY = 3
_tts_semaphore = asyncio.Semaphore(_TTS_MAX_CONCURRENCY)


class TextToSpeechService:
    def __init__(self, output_dir: Path, english_voice: str, urdu_voice: str) -> None:
        self.output_dir = output_dir
        self.english_voice = english_voice
        self.urdu_voice = urdu_voice

    async def synthesize(self, text: str, language: str) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output = self.output_dir / f"{uuid4()}.mp3"
        last_error: Exception | None = None
        voice = voice_for(
            language,
            english_voice=self.english_voice,
            urdu_voice=self.urdu_voice,
        )
        logger.info("tts voice=%s language=%s", voice, language)
        if voice == self.urdu_voice:
            text = transliterate_loanwords_for_urdu(text)
        async with _tts_semaphore:
            for attempt in range(_TTS_RETRIES):
                try:
                    await asyncio.wait_for(
                        edge_tts.Communicate(text, voice).save(str(output)),
                        timeout=_TTS_TIMEOUT,
                    )
                    return output
                except Exception as exc:
                    # Retry transient rate-limit / service hiccups with jittered backoff.
                    last_error = exc
                    output.unlink(missing_ok=True)
                    if attempt < _TTS_RETRIES - 1:
                        backoff = _TTS_BASE_BACKOFF * (attempt + 1) + random.uniform(0, 0.3)
                        await asyncio.sleep(backoff)
        raise last_error or RuntimeError("TTS synthesis failed")
