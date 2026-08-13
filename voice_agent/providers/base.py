"""Provider-neutral streaming contracts for the voice agent."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from voice_agent.profiles import UnverifiedProfileError, VerifiedVoiceProfile, VoiceProfile


class ProviderError(RuntimeError):
    """A provider failed in a way the session can report safely."""


@dataclass(frozen=True)
class TranscriptDelta:
    """An incremental speech-to-text update."""

    text: str
    is_final: bool = False
    confidence: float | None = None


@dataclass(frozen=True)
class TranscriptionResult:
    """The completed transcription metadata for a streamed utterance."""

    text: str
    language: str
    completed_at: datetime


@runtime_checkable
class STTProvider(Protocol):
    def transcribe_stream(
        self, frames: AsyncIterator[bytes], language: str
    ) -> AsyncIterator[TranscriptDelta]: ...


@runtime_checkable
class LLMProvider(Protocol):
    def stream_response(
        self,
        messages: Sequence[Mapping[str, str]],
        system_prompt: str,
    ) -> AsyncIterator[str]: ...


@runtime_checkable
class TTSProvider(Protocol):
    def synthesize_stream(
        self,
        text_chunks: AsyncIterator[str],
        profile: VerifiedVoiceProfile,
        language: str,
    ) -> AsyncIterator[bytes]: ...


def require_verified_profile(profile: VerifiedVoiceProfile) -> VoiceProfile:
    """Enforce the TTS authorization boundary at concrete provider entrypoints."""
    if not isinstance(profile, VerifiedVoiceProfile):
        raise UnverifiedProfileError("TTS requires a store-authorized voice profile")
    return profile.profile
