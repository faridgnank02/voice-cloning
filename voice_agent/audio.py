"""Audio format validation and lightweight energy-based turn detection."""

from __future__ import annotations

import math
import re
import struct
import wave
from dataclasses import dataclass


@dataclass(frozen=True)
class AudioFormat:
    sample_rate: int
    channels: int
    sample_width: int
    encoding: str


@dataclass(frozen=True)
class PCMFrame:
    data: bytes
    sample_rate: int
    channels: int
    sample_width: int = 2


@dataclass(frozen=True)
class AudioEvent:
    type: str


def decode_browser_chunk(chunk: bytes, content_type: str, max_bytes: int = 512_000) -> PCMFrame:
    if len(chunk) > max_bytes:
        raise ValueError("audio frame limit exceeded")
    if content_type.startswith("audio/pcm"):
        match = re.search(r"rate=(\d+)", content_type)
        channels = int(re.search(r"channels=(\d+)", content_type).group(1)) if "channels=" in content_type else 1
        sample_rate = int(match.group(1)) if match else 16_000
        if len(chunk) % 2:
            raise ValueError("PCM frame must contain complete samples")
        return PCMFrame(chunk, sample_rate, channels)
    if content_type == "audio/wav":
        try:
            with wave.open(__import__("io").BytesIO(chunk), "rb") as source:
                if source.getsampwidth() != 2:
                    raise ValueError("only 16-bit WAV is supported")
                return PCMFrame(source.readframes(source.getnframes()), source.getframerate(), source.getnchannels())
        except (wave.Error, EOFError) as error:
            raise ValueError("malformed WAV audio") from error
    raise ValueError("unsupported audio format")


class AudioTurnAssembler:
    def __init__(self, format: AudioFormat, max_bytes: int = 2_000_000, silence_ms: int = 600, energy_threshold: int = 300):
        self.format = format
        self.max_bytes = max_bytes
        self.silence_ms = silence_ms
        self.energy_threshold = energy_threshold
        self._buffer = bytearray()
        self._speaking = False
        self._silence_ms = 0

    def push(self, frame: bytes) -> list[AudioEvent]:
        if len(self._buffer) + len(frame) > self.max_bytes:
            raise ValueError("turn audio limit exceeded")
        if len(frame) % self.format.sample_width:
            raise ValueError("audio frame has incomplete samples")
        self._buffer.extend(frame)
        samples = struct.unpack("<" + "h" * (len(frame) // 2), frame)
        rms = math.sqrt(sum(sample * sample for sample in samples) / max(1, len(samples)))
        duration_ms = len(samples) / self.format.sample_rate * 1000
        events: list[AudioEvent] = []
        if rms >= self.energy_threshold:
            self._silence_ms = 0
            if not self._speaking:
                self._speaking = True
                events.append(AudioEvent("speech_started"))
        elif self._speaking:
            self._silence_ms += duration_ms
            if self._silence_ms >= self.silence_ms:
                self._speaking = False
                events.append(AudioEvent("speech_stopped"))
        return events

    def flush(self) -> bytes:
        result = bytes(self._buffer)
        self.reset()
        return result

    def reset(self) -> None:
        self._buffer.clear()
        self._speaking = False
        self._silence_ms = 0
