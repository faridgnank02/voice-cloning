from datetime import datetime, timezone
from pathlib import Path
import struct

from fastapi.testclient import TestClient

from voice_agent.profiles import ProfileStore, VoiceProfile
from voice_agent.server import create_app
from voice_agent.session import ConversationConfig


class AllowAllConsent:
    def verify(self, profile):
        return True


class STT:
    async def transcribe_stream(self, frames, language):
        yield type("Delta", (), {"text": "hello", "is_final": True})()


class LLM:
    async def stream_response(self, messages, system_prompt):
        yield "hi"


class TTS:
    async def synthesize_stream(self, text_chunks, profile, language):
        yield b"wav"


def setup_app(tmp_path: Path):
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"voice")
    profile = VoiceProfile("p1", "c1", datetime.now(timezone.utc), "en", str(audio))
    store = ProfileStore(AllowAllConsent())
    store.register(profile)
    return create_app(store, lambda request: (STT(), LLM(), TTS()))


def test_websocket_rejects_unverified_profile(tmp_path):
    client = TestClient(setup_app(tmp_path))
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "start_session", "voice_profile_id": "missing", "language": "en"})
        message = websocket.receive_json()
        assert message["type"] == "error"


def test_websocket_accepts_verified_profile_and_streams_turn(tmp_path):
    client = TestClient(setup_app(tmp_path))
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "start_session", "voice_profile_id": "p1", "language": "en"})
        started = websocket.receive_json()
        assert started["type"] == "session_started"
        websocket.send_bytes(struct.pack("<100h", *([1000] * 100)))
        websocket.send_json({"type": "finish_turn"})
        events = [websocket.receive_json() for _ in range(5)]
        assert any(event["type"] == "transcript_delta" for event in events)
        assert any(event["type"] == "response_audio" for event in events)
