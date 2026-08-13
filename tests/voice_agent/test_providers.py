import asyncio
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from voice_agent.profiles import ProfileStore, VerifiedVoiceProfile, VoiceProfile
from voice_agent.providers.local import LocalWhisperProvider, LocalXTTSProvider, OllamaProvider
from voice_agent.providers.openai_compatible import OpenAICompatibleProvider


class AllowAllConsent:
    def verify(self, profile: VoiceProfile) -> bool:
        return True


def make_verified_profile(tmp_path: Path) -> VerifiedVoiceProfile:
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"audio")
    profile = VoiceProfile(
        profile_id="profile-1",
        consent_id="consent-1",
        consented_at=datetime.now(timezone.utc),
        language="en",
        reference_audio_path=str(reference),
    )
    store = ProfileStore(AllowAllConsent())
    store.register(profile)
    return store.assert_usable(profile.profile_id, "en")


def test_local_whisper_adapter_delegates_buffered_audio(monkeypatch, tmp_path: Path):
    calls = {}

    def fake_run_asr(path, model_id, device_pref, language):
        calls.update(path=path, model_id=model_id, device_pref=device_pref, language=language)
        assert Path(path).read_bytes() == b"frame-1frame-2"
        return "hello"

    monkeypatch.setattr("voice_agent.providers.local.run_asr", fake_run_asr)
    provider = LocalWhisperProvider("whisper-base", "cpu", temp_dir=tmp_path)

    async def frames():
        yield b"frame-1"
        yield b"frame-2"

    result = asyncio.run(collect(provider.transcribe_stream(frames(), "fr")))

    assert result[0].text == "hello"
    assert result[0].is_final is True
    assert calls["language"] == "fr"


def test_local_whisper_adapter_surfaces_asr_exception(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("voice_agent.providers.local.run_asr", lambda *args: RuntimeError("asr failed"))
    provider = LocalWhisperProvider("whisper-base", "cpu", temp_dir=tmp_path)

    async def frames():
        yield b"audio"

    with pytest.raises(RuntimeError, match="asr failed"):
        asyncio.run(collect(provider.transcribe_stream(frames(), "en")))


def test_local_xtts_adapter_rejects_unverified_profile(monkeypatch, tmp_path: Path):
    called = False

    def fake_tts(*args, **kwargs):
        nonlocal called
        called = True
        return 22050, np.zeros(4, dtype=np.float32)

    monkeypatch.setattr("voice_agent.providers.local.run_tts_clone", fake_tts)
    provider = LocalXTTSProvider()
    raw_profile = make_verified_profile(tmp_path).profile

    async def text_chunks():
        yield "hello"

    with pytest.raises(TypeError, match="VerifiedVoiceProfile"):
        asyncio.run(collect(provider.synthesize_stream(text_chunks(), raw_profile, "en")))
    assert called is False


def test_local_xtts_adapter_returns_wav_bytes(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "voice_agent.providers.local.run_tts_clone",
        lambda *args, **kwargs: (22050, np.zeros(8, dtype=np.float32)),
    )
    provider = LocalXTTSProvider()
    profile = make_verified_profile(tmp_path)

    async def text_chunks():
        yield "hello"
        yield " world"

    output = asyncio.run(collect(provider.synthesize_stream(text_chunks(), profile, "en")))

    assert len(output) == 1
    assert output[0].startswith(b"RIFF")


def test_ollama_adapter_parses_ndjson_deltas():
    provider = OllamaProvider("http://ollama", "llama3")
    async def lines(**kwargs):
        yield json.dumps({"message": {"content": "Hel"}})
        yield json.dumps({"message": {"content": "lo"}, "done": True})
    provider._stream_lines = lines

    async def collect_response():
        return [item async for item in provider.stream_response([], "system")]

    assert asyncio.run(collect_response()) == ["Hel", "lo"]


def test_ollama_adapter_rejects_malformed_ndjson():
    provider = OllamaProvider("http://ollama", "llama3")
    async def lines(**kwargs):
        yield "not-json"
    provider._stream_lines = lines

    async def collect_response():
        return [item async for item in provider.stream_response([], "system")]

    with pytest.raises(ValueError, match="malformed Ollama"):
        asyncio.run(collect_response())


def test_openai_compatible_adapter_parses_sse_deltas():
    provider = OpenAICompatibleProvider("http://api", "secret", "model")
    async def lines(**kwargs):
        yield 'data: {"choices":[{"delta":{"content":"Hi"}}]}'
        yield 'data: {"choices":[{"delta":{"content":"!"}}]}'
        yield "data: [DONE]"
    provider._stream_lines = lines

    async def collect_response():
        return [item async for item in provider.stream_response([], "system")]

    assert asyncio.run(collect_response()) == ["Hi", "!"]


def test_openai_compatible_adapter_does_not_expose_api_key():
    provider = OpenAICompatibleProvider("http://api", "secret", "model")
    assert "secret" not in repr(provider)


async def collect(stream):
    return [item async for item in stream]
