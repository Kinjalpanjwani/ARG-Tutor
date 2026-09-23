from pathlib import Path

import pytest

from app.speech.tts import TextToSpeechService


@pytest.fixture
def voices():
    return []


class FakeCommunicate:
    def __init__(self, text, voice, voices):
        self.text = text
        self.voice = voice
        self.voices = voices

    async def save(self, out):
        self.voices.append(self.voice)
        return None


def _install_fake_communicate(monkeypatch, voices):
    import edge_tts

    def fake_communicate(text, voice):
        return FakeCommunicate(text, voice, voices)

    monkeypatch.setattr(edge_tts, "Communicate", fake_communicate)


@pytest.fixture
def service(tmp_path):
    return TextToSpeechService(
        Path(tmp_path), "en-US-AriaNeural", "ur-PK-AsadNeural"
    )


@pytest.mark.asyncio
async def test_synthesize_uses_english_voice_for_english(monkeypatch, service, voices):
    _install_fake_communicate(monkeypatch, voices)
    path = await service.synthesize("hello", "English")
    assert voices == ["en-US-AriaNeural"]
    assert isinstance(path, Path)
    assert path.name.endswith(".mp3")


@pytest.mark.asyncio
async def test_synthesize_uses_urdu_voice_for_urdu(monkeypatch, service, voices):
    _install_fake_communicate(monkeypatch, voices)
    await service.synthesize("سلام", "Urdu")
    assert voices == ["ur-PK-AsadNeural"]


@pytest.mark.asyncio
async def test_synthesize_collapses_roman_urdu_to_urdu_voice(monkeypatch, service, voices):
    _install_fake_communicate(monkeypatch, voices)
    await service.synthesize("yeh theek hai", "Roman Urdu")
    assert voices == ["ur-PK-AsadNeural"]


@pytest.mark.asyncio
async def test_synthesize_falls_back_to_english_voice_for_unknown(monkeypatch, service, voices):
    _install_fake_communicate(monkeypatch, voices)
    await service.synthesize("hello", "Tagalog")
    assert voices == ["en-US-AriaNeural"]


class FlakyCommunicate:
    def __init__(self, text, voice, attempt):
        self.text = text
        self.voice = voice
        self.attempt = attempt

    async def save(self, out):
        self.attempt[0] += 1
        if self.attempt[0] < 3:
            raise RuntimeError("edge-tts rate limited")
        Path(out).write_bytes(b"audio")
        return None


def _install_flaky_communicate(monkeypatch, attempt):
    import edge_tts

    def fake_communicate(text, voice):
        return FlakyCommunicate(text, voice, attempt)

    monkeypatch.setattr(edge_tts, "Communicate", fake_communicate)


@pytest.mark.asyncio
async def test_synthesize_retries_rate_limited_attempts(monkeypatch, service):
    attempt = [0]
    _install_flaky_communicate(monkeypatch, attempt)
    path = await service.synthesize("hello", "English")
    assert attempt[0] == 3
    assert path.read_bytes() == b"audio"


@pytest.mark.asyncio
async def test_synthesize_raises_after_exhausting_retries(monkeypatch, service):
    attempt = [0]
    import edge_tts

    class FailingCommunicate:
        def __init__(self, text, voice):
            pass

        async def save(self, out):
            attempt[0] += 1
            raise RuntimeError("edge-tts down")

    monkeypatch.setattr(edge_tts, "Communicate", lambda text, voice: FailingCommunicate(text, voice))
    with pytest.raises(RuntimeError):
        await service.synthesize("hello", "English")
    assert attempt[0] == 3
    assert list(service.output_dir.iterdir()) == []