"""Measure mock voice-agent turn latency without loading speech models."""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_agent.profiles import ProfileStore, VoiceProfile
from voice_agent.session import ConversationConfig, ConversationSession


class Consent:
    def verify(self, profile):
        return True


class STT:
    async def transcribe_stream(self, frames, language):
        yield type("Delta", (), {"text": "hello", "is_final": True})()


class LLM:
    async def stream_response(self, messages, system_prompt):
        yield "hello back"


class TTS:
    async def synthesize_stream(self, text_chunks, profile, language):
        yield b"mock-audio"


async def run_once(profile):
    session = ConversationSession(
        profile,
        ConversationConfig("Be concise", "en", "mock", "mock", "mock"),
        STT(), LLM(), TTS(),
    )
    session.start_turn()
    await session.accept_audio(b"mock-input")
    started = time.perf_counter()
    await session.finish_turn()
    return (time.perf_counter() - started) * 1000


async def main(iterations: int) -> None:
    reference = Path("/tmp/voice-agent-benchmark-reference.wav")
    reference.write_bytes(b"mock")
    profile = VoiceProfile("benchmark", "benchmark-consent", datetime.now(timezone.utc), "en", str(reference))
    store = ProfileStore(Consent())
    store.register(profile)
    verified = store.assert_usable("benchmark", "en")
    samples = [await run_once(verified) for _ in range(iterations)]
    print(f"iterations={len(samples)} p50_ms={statistics.median(samples):.2f} p95_ms={sorted(samples)[max(0, int(len(samples) * .95) - 1)]:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="run the dependency-free mock benchmark")
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if not args.mock:
        parser.error("only --mock is available until model-backed benchmarks are configured")
    asyncio.run(main(args.iterations))
