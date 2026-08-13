"""FastAPI WebSocket service for consent-backed voice conversations."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Callable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from voice_agent.profiles import ConsentVerifier, ProfileError, ProfileStore, VoiceProfile
from voice_agent.config import Settings
from voice_agent.factory import build_profile_store, build_provider_factory
from voice_agent.audio import AudioFormat, AudioTurnAssembler
from voice_agent.protocol import ClientStartSession, event_json
from voice_agent.session import ConversationConfig, ConversationSession


ProviderFactory = Callable[[ClientStartSession], tuple[object, object, object]]


def _unconfigured_factory(request: ClientStartSession):
    raise RuntimeError("provider factory is not configured")


class _RejectAllConsent:
    def verify(self, profile: VoiceProfile) -> bool:
        return False


def create_app(
    profile_store: ProfileStore | None = None,
    provider_factory: ProviderFactory | None = None,
) -> FastAPI:
    app = FastAPI(title="Consent-backed Voice Agent")
    store = profile_store or ProfileStore(_RejectAllConsent())
    factory = provider_factory or _unconfigured_factory

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse("voice_agent/static/index.html")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    app.mount("/static", StaticFiles(directory="voice_agent/static"), name="static")

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        session: ConversationSession | None = None
        event_task: asyncio.Task | None = None
        session_id = str(uuid.uuid4())
        try:
            raw = await websocket.receive_json()
            if raw.get("type") != "start_session":
                await websocket.send_json(event_json("error", session_id, 0, message="first message must start a session"))
                await websocket.close(code=1008)
                return
            try:
                request = ClientStartSession(**{key: value for key, value in raw.items() if key != "type"})
                profile = store.assert_usable(request.voice_profile_id, request.language)
                stt, llm, tts = factory(request)
            except (ProfileError, RuntimeError, ValueError, TypeError) as error:
                await websocket.send_json(event_json("error", session_id, 0, message=str(error)))
                await websocket.close(code=1008)
                return
            config = ConversationConfig(
                request.system_prompt,
                request.language,
                request.stt_provider,
                request.llm_provider,
                request.tts_provider,
            )
            session = ConversationSession(profile, config, stt, llm, tts)
            assembler = AudioTurnAssembler(AudioFormat(16000, 1, 2, "pcm_s16le"))

            async def forward_events() -> None:
                async for event in session.events_stream():
                    await websocket.send_json(event_json(event.type, session_id, event.turn_id, **event.data))

            event_task = asyncio.create_task(forward_events())
            await websocket.send_json(
                event_json("session_started", session_id, 0, audio_format={"container": "wav", "sample_rate": 22050})
            )
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    if session.active_task is None and not session._audio:
                        session.start_turn()
                    await session.accept_audio(message["bytes"])
                    for audio_event in assembler.push(message["bytes"]):
                        await websocket.send_json(event_json(audio_event.type, session_id, session._turn_id))
                        if audio_event.type == "speech_stopped":
                            asyncio.create_task(session.finish_turn())
                    continue
                payload = message.get("text")
                if payload is None:
                    continue
                command = json.loads(payload)
                if command.get("type") == "finish_turn":
                    if session._audio and session.active_task is None:
                        asyncio.create_task(session.finish_turn())
                elif command.get("type") == "interrupt":
                    await session.interrupt()
        except WebSocketDisconnect:
            pass
        finally:
            if session is not None:
                await session.close()
            if event_task is not None:
                event_task.cancel()

    return app


def create_configured_app() -> FastAPI:
    settings = Settings.from_env()
    return create_app(build_profile_store(settings), build_provider_factory(settings))


app = create_configured_app()
