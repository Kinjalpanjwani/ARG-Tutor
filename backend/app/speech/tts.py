from pathlib import Path
from uuid import uuid4
import asyncio

import edge_tts


class TextToSpeechService:
    def __init__(self, output_dir: Path, english_voice: str, urdu_voice: str) -> None:
        self.output_dir = output_dir
        self.voices = {"English": english_voice, "Urdu": urdu_voice, "Roman Urdu": urdu_voice}

    async def synthesize(self, text: str, language: str) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output = self.output_dir / f"{uuid4()}.mp3"
        try:
            await asyncio.wait_for(
                edge_tts.Communicate(
                    text, self.voices.get(language, self.voices["English"])
                ).save(str(output)),
                timeout=30,
            )
        except Exception:
            output.unlink(missing_ok=True)
            raise
        return output
