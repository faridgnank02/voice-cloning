import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest

from voice_agent.profiles import ProfileStore, VoiceProfile
from voice_agent.providers.base import TranscriptDelta
from voice_agent.session import ConversationConfig, ConversationSession, SessionEvent


class AllowAllConsent:
    def verify(self, profile):
        return True


def verified_profile(tmp_path: Path):
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"voice")
    profile = VoiceProfile("p1", "c1", datetime.now(timezone.utc), "en", str(audio))
    store = ProfileStore(AllowAllConsent())
    store.register(profile)
    return store.assert_usable("p1", "en")


class FakeSTT:
    async def transcribe_stream(self, frames, language):
        self.frames = b"".join([frame async for frame in frames])
        yield TranscriptDelta("hello", is_final=True)


class FakeLLM:
    async def stream_response(self, messages, system_prompt):
        self.messages = messages
        yield "Hi"
        yield " there"


class FakeTTS:
    async def synthesize_stream(self, text_chunks, profile, language):
        self.profile = profile
        self.text = ""
        async for chunk in text_chunks:
            self.text += chunk
        yield b"audio-1"
        yield b"audio-2"


class FailingLLM:
    async def stream_response(self, messages, system_prompt):
        raise RuntimeError("llm down")
        yield "unreachable"


class BlockingLLM:
    def __init__(self):
        self.started = asyncio.Event()
        self.cancelled = False

    async def stream_response(self, messages, system_prompt):
        self.started.set()
        try:
            yield "partial"
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            self.cancelled = True
            raise


def make_session(tmp_path, llm=None, max_history=20):
    return ConversationSession(
        profile=verified_profile(tmp_path),
        config=ConversationConfig("Be concise", "en", "stt", "llm", "tts", max_history),
        stt=FakeSTT(),
        llm=llm or FakeLLM(),
        tts=FakeTTS(),
    )


@pytest.mark.asyncio
async def test_complete_turn_emits_transcript_text_audio_and_completion(tmp_path):
    session = make_session(tmp_path)
    turn_id = session.start_turn()
    await session.accept_audio(b"audio")
    await session.finish_turn()

    assert turn_id == 1
    assert [event.type for event in session.events] == [
        "transcript_delta", "assistant_text_delta", "assistant_text_delta",
        "response_audio", "response_audio", "turn_completed", "metrics",
    ]
    assert session.events[0].data["text"] == "hello"
    assert session.events[2].data["text"] == " there"
    assert session.events[3].data["audio"] == b"audio-1"


@pytest.mark.asyncio
async def test_audio_buffer_is_bounded(tmp_path):
    session = make_session(tmp_path)
    session.max_audio_bytes = 4
    session.start_turn()

    with pytest.raises(ValueError, match="audio buffer limit"):
        await session.accept_audio(b"12345")


@pytest.mark.asyncio
async def test_history_is_truncated_to_configured_limit(tmp_path):
    session = make_session(tmp_path, max_history=2)
    for _ in range(3):
        session.start_turn()
        await session.accept_audio(b"audio")
        await session.finish_turn()

    assert len(session.history) == 2
    assert session.history[-1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_provider_error_emits_safe_error_and_releases_task(tmp_path):
    session = make_session(tmp_path, llm=FailingLLM())
    session.start_turn()
    await session.accept_audio(b"audio")
    await session.finish_turn()

    assert session.events[-1].type == "error"
    assert session.events[-1].data == {"stage": "llm", "message": "llm down"}
    assert session.active_task is None


@pytest.mark.asyncio
async def test_interrupt_cancels_generation_and_drops_stale_audio(tmp_path):
    llm = BlockingLLM()
    session = make_session(tmp_path, llm=llm)
    session.start_turn()
    await session.accept_audio(b"audio")
    task = asyncio.create_task(session.finish_turn())
    await llm.started.wait()
    await session.interrupt()
    await task

    assert llm.cancelled is True
    assert any(event.type == "turn_interrupted" for event in session.events)
    assert not any(event.type == "response_audio" for event in session.events)


@pytest.mark.asyncio
async def test_events_stream_yields_deltas_before_turn_finishes(tmp_path):
    session = make_session(tmp_path)
    session.start_turn()
    await session.accept_audio(b"audio")
    task = asyncio.create_task(session.finish_turn())
    stream = session.events_stream()
    first = await asyncio.wait_for(anext(stream), timeout=1)
    await task

    assert first.type == "transcript_delta"
