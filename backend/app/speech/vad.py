from pathlib import Path
from threading import Lock


class VoiceActivityDetector:
    _model = None
    _lock = Lock()

    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from silero_vad import load_silero_vad

                    type(self)._model = load_silero_vad()
        return self._model

    def contains_speech(self, wav_path: Path, min_speech_ms: int = 300) -> bool:
        import soundfile as sf
        import torch
        from silero_vad import get_speech_timestamps

        samples, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if sample_rate != 16000:
            raise ValueError("VAD audio must be 16 kHz")
        if getattr(samples, "ndim", 1) > 1:
            samples = samples.mean(axis=1)
        audio = torch.from_numpy(samples)
        return bool(get_speech_timestamps(
            audio, self.model, sampling_rate=16000,
            min_speech_duration_ms=min_speech_ms,
        ))
