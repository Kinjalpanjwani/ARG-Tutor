from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile

from app.llm.groq_client import GroqService
from app.speech.vad import VoiceActivityDetector


class SpeechToTextService:
    def __init__(self, groq: GroqService, vad: VoiceActivityDetector) -> None:
        self.groq = groq
        self.vad = vad

    async def transcribe(self, path: Path) -> tuple[bool, str]:
        with NamedTemporaryFile(suffix=".wav", delete=False) as temporary:
            wav_path = Path(temporary.name)
        try:
            process = subprocess.run(
                ["ffmpeg", "-y", "-i", str(path), "-ac", "1", "-ar", "16000", str(wav_path)],
                capture_output=True,
            )
            if process.returncode != 0:
                raise ValueError("Audio could not be decoded")
            if not self.vad.contains_speech(wav_path):
                return False, ""
            # Language is intentionally omitted so Groq Whisper auto-detects it.
            text = await self.groq.transcribe(str(wav_path))
            return bool(text), text
        finally:
            wav_path.unlink(missing_ok=True)
