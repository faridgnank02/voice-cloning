"""Adapters for the repository's local Whisper, XTTS, and Ollama engines."""

from __future__ import annotations

import asyncio
import io
import json
import tempfile
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path

import httpx
import numpy as np
from scipy.io import wavfile

from voice_agent.profiles import UnverifiedProfileError, VerifiedVoiceProfile
from voice_agent.providers.base import ProviderError, TranscriptDelta, require_verified_profile


def run_asr(*args, **kwargs):
    from src.process import run_asr as implementation

    return implementation(*args, **kwargs)


def run_tts_clone(*args, **kwargs):
    from src.tts import run_tts_clone as implementation

    return implementation(*args, **kwargs)


class LocalWhisperProvider:
    def __init__(self, model_id: str, device_preference: str, temp_dir: str | Path | None = None):
        self.model_id = model_id
        self.device_preference = device_preference
        self.temp_dir = Path(temp_dir) if temp_dir else None

    async def transcribe_stream(
        self, frames: AsyncIterator[bytes], language: str
    ) -> AsyncIterator[TranscriptDelta]:
        data = bytearray()
        async for frame in frames:
            data.extend(frame)
        with tempfile.NamedTemporaryFile(dir=self.temp_dir, suffix=".audio", delete=False) as handle:
            handle.write(data)
            path = handle.name
        try:
            result = await asyncio.to_thread(
                run_asr, path, self.model_id, self.device_preference, language
            )
        finally:
            Path(path).unlink(missing_ok=True)
        if isinstance(result, Exception):
            raise result
        yield TranscriptDelta(text=result, is_final=True)


class LocalXTTSProvider:
    def __init__(self, model_id: str = "coqui/XTTS-v2"):
        self.model_id = model_id

    async def synthesize_stream(
        self,
        text_chunks: AsyncIterator[str],
        profile: VerifiedVoiceProfile,
        language: str,
    ) -> AsyncIterator[bytes]:
        if not isinstance(profile, VerifiedVoiceProfile):
            raise TypeError("TTS requires VerifiedVoiceProfile")
        authorized = require_verified_profile(profile)
        text = ""
        async for chunk in text_chunks:
            text += chunk
        result = await asyncio.to_thread(
            run_tts_clone,
            authorized.reference_audio_path,
            text,
            self.model_id,
            language,
        )
        if isinstance(result, Exception):
            raise result
        sample_rate, waveform = result
        output = io.BytesIO()
        wavfile.write(output, sample_rate, np.asarray(waveform))
        yield output.getvalue()


class OllamaProvider:
    def __init__(self, base_url: str, model: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    async def stream_response(
        self,
        messages: Sequence[Mapping[str, str]],
        system_prompt: str,
    ) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "stream": True,
            "messages": [{"role": "system", "content": system_prompt}, *messages],
        }
        async for line in self._stream_lines(payload=payload):
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("malformed Ollama response") from error
            try:
                content = item["message"]["content"]
            except (KeyError, TypeError) as error:
                raise ValueError("malformed Ollama response") from error
            if content:
                yield content

    async def _stream_lines(self, *, payload: dict) -> AsyncIterator[str]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        yield line
        except httpx.HTTPError as error:
            raise ProviderError("Ollama request failed") from error
