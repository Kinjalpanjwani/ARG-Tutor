import logging
from pathlib import Path
import shutil
import subprocess
from tempfile import NamedTemporaryFile

from app.llm.groq_client import GroqService
from app.speech.vad import VoiceActivityDetector

logger = logging.getLogger(__name__)

# Map the app's lesson language names to Whisper ISO-639-1 codes. Forcing a code
# stops Whisper from auto-detecting Urdu/Hindustani speech as Hindi.
_LANGUAGE_TO_WHISPER_CODE = {
    "urdu": "ur",
    "roman urdu": "ur",
    "english": "en",
}


def _whisper_code(language: str | None) -> str | None:
    if not language:
        return None
    return _LANGUAGE_TO_WHISPER_CODE.get(language.strip().lower())


def _get_ffmpeg_executable() -> str:
    """Find ffmpeg on PATH or fall back to imageio-ffmpeg bundled binary."""
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError(f"ffmpeg is not installed and imageio-ffmpeg executable could not be resolved: {exc}") from exc


class SpeechToTextService:
    def __init__(self, groq: GroqService, vad: VoiceActivityDetector) -> None:
        self.groq = groq
        self.vad = vad

    async def transcribe(self, path: Path, language: str | None = None) -> tuple[bool, str]:
        code = _whisper_code(language)
        # 1. Attempt transcription directly on raw audio file
        try:
            text = await self.groq.transcribe(str(path), language=code) if code else await self.groq.transcribe(str(path))
            return bool(text), text
        except Exception as direct_exc:
            logger.warning("[STT] Direct raw-audio transcription failed, falling back to ffmpeg: %s", direct_exc)

        # 2. Fallback: Convert to 16kHz mono WAV using ffmpeg (bundled or system)
        with NamedTemporaryFile(suffix=".wav", delete=False) as temporary:
            wav_path = Path(temporary.name)
        try:
            ffmpeg_exe = _get_ffmpeg_executable()
            process = subprocess.run(
                [ffmpeg_exe, "-y", "-i", str(path), "-ac", "1", "-ar", "16000", str(wav_path)],
                capture_output=True,
            )
            if process.returncode != 0:
                stderr = process.stderr.decode(errors="replace")
                raise ValueError(f"Audio conversion failed: {stderr}")

            # Optional VAD gating on valid wav
            try:
                if not self.vad.contains_speech(wav_path):
                    return False, ""
            except Exception as vad_exc:
                logger.warning("[STT] VAD check skipped due to error: %s", vad_exc)

            text = await self.groq.transcribe(str(wav_path), language=code) if code else await self.groq.transcribe(str(wav_path))
            return bool(text), text
        finally:
            wav_path.unlink(missing_ok=True)
