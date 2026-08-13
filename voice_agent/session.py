"""Conversation turn orchestration with bounded state and cancellation."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass

from voice_agent.profiles import VerifiedVoiceProfile
from voice_agent.providers.base import LLMProvider, STTProvider, TTSProvider


@dataclass(frozen=True)
class ConversationConfig:
    system_prompt: str
    language: str
    stt_provider: str
    llm_provider: str
    tts_provider: str
    max_history: int = 20


@dataclass(frozen=True)
class SessionEvent:
    type: str
    turn_id: int
    data: dict


class ConversationSession:
    def __init__(
        self,
        profile: VerifiedVoiceProfile,
        config: ConversationConfig,
        stt: STTProvider,
        llm: LLMProvider,
        tts: TTSProvider,
        max_audio_bytes: int = 2_000_000,
    ) -> None:
        self.profile = profile
        self.config = config
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.max_audio_bytes = max_audio_bytes
        self.events: list[SessionEvent] = []
        self._event_queue: asyncio.Queue[SessionEvent | None] = asyncio.Queue(maxsize=128)
        self.history: list[dict[str, str]] = []
        self._audio = bytearray()
        self._turn_id = 0
        self._active_task: asyncio.Task | None = None
        self._interrupted: set[int] = set()
        self._closed = False

    @property
    def active_task(self) -> asyncio.Task | None:
        return self._active_task

    def start_turn(self) -> int:
        if self._closed:
            raise RuntimeError("session is closed")
        self._turn_id += 1
        self._audio.clear()
        return self._turn_id

    async def accept_audio(self, frame: bytes) -> None:
        if self._closed:
            raise RuntimeError("session is closed")
        if len(self._audio) + len(frame) > self.max_audio_bytes:
            raise ValueError("audio buffer limit exceeded")
        self._audio.extend(frame)

    async def events_stream(self) -> AsyncIterator[SessionEvent]:
        while True:
            event = await self._event_queue.get()
            if event is None:
                return
            yield event

    async def finish_turn(self) -> None:
        if self._closed:
            raise RuntimeError("session is closed")
        turn_id = self._turn_id
        audio = bytes(self._audio)
        self._audio.clear()
        current = asyncio.current_task()
        self._active_task = current
        started = time.perf_counter()
        try:
            async def frames() -> AsyncIterator[bytes]:
                yield audio

            transcript = ""
            async for delta in self.stt.transcribe_stream(frames(), self.config.language):
                transcript += delta.text
                self._emit("transcript_delta", turn_id, {"text": delta.text, "final": delta.is_final})
            self._append_history("user", transcript)

            response = ""
            async for chunk in self.llm.stream_response(
                self._history_for_provider(), self.config.system_prompt
            ):
                response += chunk
                self._emit("assistant_text_delta", turn_id, {"text": chunk})
            self._append_history("assistant", response)

            async def text_chunks() -> AsyncIterator[str]:
                yield response

            async for audio_chunk in self.tts.synthesize_stream(
                text_chunks(), self.profile, self.config.language
            ):
                if turn_id != self._turn_id or turn_id in self._interrupted:
                    continue
                self._emit("response_audio", turn_id, {"audio": audio_chunk})
            self._emit("turn_completed", turn_id, {})
            self._emit(
                "metrics", turn_id, {"total_ms": (time.perf_counter() - started) * 1000}
            )
        except asyncio.CancelledError:
            if turn_id not in self._interrupted:
                raise
        except Exception as error:
            self._emit("error", turn_id, {"stage": self._stage_for_error(error), "message": str(error)})
        finally:
            if self._active_task is current:
                self._active_task = None

    async def interrupt(self) -> None:
        task = self._active_task
        if task is None or task.done():
            return
        turn_id = self._turn_id
        self._interrupted.add(turn_id)
        self._emit("turn_interrupted", turn_id, {})
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            self._active_task = None

    async def close(self) -> None:
        self._closed = True
        await self.interrupt()
        self._audio.clear()
        if not self._event_queue.full():
            self._event_queue.put_nowait(None)

    def _append_history(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        self.history = self.history[-self.config.max_history :]

    def _history_for_provider(self) -> Sequence[Mapping[str, str]]:
        return tuple(self.history)

    def _emit(self, event_type: str, turn_id: int, data: dict) -> None:
        event = SessionEvent(event_type, turn_id, data)
        self.events.append(event)
        if not self._event_queue.full():
            self._event_queue.put_nowait(event)

    @staticmethod
    def _stage_for_error(error: Exception) -> str:
        name = error.__class__.__name__.lower()
        if "stt" in name or "transcri" in name:
            return "stt"
        if "tts" in name or "synth" in name:
            return "tts"
        return "llm"
