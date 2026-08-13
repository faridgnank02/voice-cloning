from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import get_type_hints

import pytest

import voice_agent.profiles as profiles
import voice_agent.providers.base as provider_contracts
from voice_agent.profiles import VoiceProfile
from voice_agent.providers.base import (
    LLMProvider,
    STTProvider,
    TTSProvider,
    TranscriptDelta,
    TranscriptionResult,
)


class FakeSTT:
    async def transcribe_stream(
        self, frames: AsyncIterator[bytes], language: str
    ) -> AsyncIterator[TranscriptDelta]:
        yield TranscriptDelta(text="hello", is_final=True)


class FakeLLM:
    async def stream_response(
        self, messages: list[dict[str, str]], system_prompt: str
    ) -> AsyncIterator[str]:
        yield "hello"


class FakeTTS:
    async def synthesize_stream(
        self,
        text_chunks: AsyncIterator[str],
        profile: "VerifiedVoiceProfile",
        language: str,
    ) -> AsyncIterator[bytes]:
        yield b"audio"


def test_provider_protocols_are_runtime_checkable() -> None:
    assert isinstance(FakeSTT(), STTProvider)
    assert isinstance(FakeLLM(), LLMProvider)
    assert isinstance(FakeTTS(), TTSProvider)


def test_verified_profile_capability_rejects_a_bare_profile() -> None:
    profile = VoiceProfile(
        profile_id="profile-1",
        consent_id="consent-1",
        consented_at=datetime(2026, 7, 18, tzinfo=timezone.utc),
        language="en",
        reference_audio_path="reference.wav",
    )

    assert hasattr(profiles, "VerifiedVoiceProfile")
    with pytest.raises(profiles.UnverifiedProfileError):
        profiles.VerifiedVoiceProfile(profile)

    assert hasattr(provider_contracts, "require_verified_profile")
    with pytest.raises(profiles.UnverifiedProfileError):
        provider_contracts.require_verified_profile(profile)


def test_tts_contract_requires_a_verified_profile_capability() -> None:
    annotations = get_type_hints(TTSProvider.synthesize_stream)

    assert annotations["profile"] is profiles.VerifiedVoiceProfile


def test_transcription_types_hold_streaming_and_final_values() -> None:
    delta = TranscriptDelta(text="partial", is_final=False, confidence=0.8)
    result = TranscriptionResult(
        text="final",
        language="en",
        completed_at=datetime(2026, 7, 18, tzinfo=timezone.utc),
    )

    assert delta.text == "partial"
    assert not delta.is_final
    assert result.text == "final"
    assert result.language == "en"
