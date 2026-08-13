"""Wire-level WebSocket event models and JSON-safe serialization."""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ClientStartSession:
    voice_profile_id: str
    language: str
    stt_provider: str = "local"
    llm_provider: str = "ollama"
    tts_provider: str = "local"
    system_prompt: str = "You are a helpful voice assistant."


def event_json(event_type: str, session_id: str, turn_id: int, **data: Any) -> dict[str, Any]:
    return {"type": event_type, "session_id": session_id, "turn_id": turn_id, **_json_safe(data)}


def _json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"encoding": "base64", "data": base64.b64encode(value).decode("ascii")}
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
