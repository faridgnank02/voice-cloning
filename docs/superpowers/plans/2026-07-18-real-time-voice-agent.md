# Real-Time Voice Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a consent-backed, provider-agnostic browser voice agent with WebSocket streaming, interruption, local model support, and measurable latency while preserving the existing Gradio application.

**Architecture:** Add a focused `voice_agent/` package containing profile validation, provider protocols/adapters, session state, and WebSocket events. A small FastAPI service will coordinate browser audio, VAD, streaming STT/LLM/TTS, and cancellation; a static browser client will handle capture and playback.

**Tech Stack:** Python 3.10, FastAPI/Uvicorn, WebSocket, browser MediaRecorder/Web Audio APIs, Whisper through the existing Transformers integration, XTTS v2 through `src.tts`, Ollama/OpenAI-compatible HTTP adapters, pytest, and Playwright or browser-level smoke tests where available.

## Global Constraints

- Mandatory consent verification before a voice profile can be registered or used.
- Provider selection must be configurable without changing conversation logic.
- Audio is ephemeral by default; durable storage is opt-in and outside the MVP.
- Credentials remain server-side.
- TTS must never fall back to an unverified voice.
- Transcript history and audio buffers have explicit bounds.
- Sub-500 ms is an optimization target, not an initial guarantee with XTTS v2.
- Existing Gradio consent and cloning behavior must continue working.

---

## File map

- Create `voice_agent/profiles.py`: verified voice-profile data and authorization checks.
- Create `voice_agent/providers/base.py`: provider protocols and streaming result types.
- Create `voice_agent/providers/local.py`: adapters around existing Whisper and XTTS functions plus Ollama.
- Create `voice_agent/providers/openai_compatible.py`: configurable OpenAI-compatible LLM adapter.
- Create `voice_agent/session.py`: turn state, bounded history, cancellation, and orchestration.
- Create `voice_agent/protocol.py`: WebSocket event models and serialization.
- Create `voice_agent/server.py`: FastAPI application and WebSocket endpoint.
- Create `voice_agent/static/index.html`, `client.js`, and `styles.css`: browser capture, playback, controls, and event rendering.
- Create `tests/voice_agent/`: unit and integration tests using fake providers.
- Modify `requirements.txt`: add server, VAD, and test dependencies with compatible minimums.
- Modify `README.md`: document consent-backed voice-agent startup and provider configuration.
- Create `scripts/benchmark_voice_agent.py`: stage timing and p50/p95 report.

### Task 1: Add profile and provider contracts

**Files:**
- Create: `voice_agent/profiles.py`
- Create: `voice_agent/providers/base.py`
- Create: `tests/voice_agent/test_profiles.py`
- Create: `tests/voice_agent/test_provider_contracts.py`

**Interfaces:**
- `VoiceProfile(profile_id: str, consent_id: str, consented_at: datetime, language: str, reference_audio_path: str, revoked: bool = False)`
- `ProfileStore.register(profile)`, `ProfileStore.get(profile_id)`, and `ProfileStore.assert_usable(profile_id, language)`
- `STTProvider.transcribe_stream(frames, language) -> AsyncIterator[TranscriptDelta]`
- `LLMProvider.stream_response(messages, system_prompt) -> AsyncIterator[str]`
- `TTSProvider.synthesize_stream(text_chunks, profile, language) -> AsyncIterator[bytes]`

- [ ] Write failing tests for valid registration, missing profile, revoked profile, language mismatch, and missing reference audio.
- [ ] Run `pytest tests/voice_agent/test_profiles.py -v`; expect failures because the package does not exist.
- [ ] Implement in-memory profile storage and signed-profile serialization using standard-library HMAC; never expose reference audio through a client response.
- [ ] Add runtime-checkable provider protocols and typed delta/result dataclasses.
- [ ] Run both test files; expect all contract and authorization tests to pass.

### Task 2: Implement provider adapters

**Files:**
- Create: `voice_agent/providers/local.py`
- Create: `voice_agent/providers/openai_compatible.py`
- Create: `tests/voice_agent/test_providers.py`

**Interfaces:**
- `LocalWhisperProvider(model_id, device_preference)` delegates batch fallback to `src.process.run_asr`.
- `LocalXTTSProvider(model_id)` delegates synthesis to `src.tts.run_tts_clone` and rejects invalid profiles before calling it.
- `OllamaProvider(base_url, model)` streams newline-delimited responses and raises a typed provider error on non-2xx responses.
- `OpenAICompatibleProvider(base_url, api_key, model)` streams chat-completion deltas without sending credentials to application code outside the adapter.

- [ ] Write fake HTTP-stream tests for Ollama and OpenAI-compatible delta parsing, malformed chunks, HTTP errors, and cancellation.
- [ ] Run `pytest tests/voice_agent/test_providers.py -v`; expect failures.
- [ ] Implement adapters with async cancellation and provider-neutral output types.
- [ ] Add local adapter tests that monkeypatch `src.process.run_asr` and `src.tts.run_tts_clone` so tests never download or load models.
- [ ] Run provider tests; expect all mocked tests to pass.

### Task 3: Implement session orchestration and cancellation

**Files:**
- Create: `voice_agent/session.py`
- Create: `tests/voice_agent/test_session.py`

**Interfaces:**
- `ConversationConfig(system_prompt: str, language: str, stt_provider: str, llm_provider: str, tts_provider: str, max_history: int = 20)`
- `ConversationSession.start_turn() -> int`, `accept_audio(frame: bytes)`, `finish_turn()`, `interrupt()`, and `close()`.
- Session emits typed events for speech, transcript, assistant text, audio, completion, interruption, and metrics.

- [ ] Write fake-provider tests for complete turn ordering, bounded history, provider error events, disconnect cleanup, and interruption dropping stale audio.
- [ ] Run `pytest tests/voice_agent/test_session.py -v`; expect failures.
- [ ] Implement an explicit per-turn cancellation task and ignore output whose `turn_id` is no longer active.
- [ ] Add bounded byte buffers and history truncation before provider calls.
- [ ] Run the session tests; expect all fake-provider scenarios to pass.

### Task 4: Define protocol and WebSocket service

**Files:**
- Create: `voice_agent/protocol.py`
- Create: `voice_agent/server.py`
- Create: `tests/voice_agent/test_protocol.py`
- Create: `tests/voice_agent/test_server.py`
- Modify: `requirements.txt`

**Interfaces:**
- JSON `start_session` request includes `voice_profile_id`, `language`, and provider/config names.
- Server replies with `session_started` containing `session_id` and negotiated audio format.
- Binary frames are accepted only after session startup; all JSON events include `session_id` and `turn_id`.

- [ ] Write serialization tests for every initial event type and rejection of malformed or unauthorized startup requests.
- [ ] Run protocol tests; expect failures.
- [ ] Implement Pydantic event models and a FastAPI WebSocket route that creates isolated sessions, forwards frames, and closes sessions in `finally`.
- [ ] Add dependencies for FastAPI, Uvicorn, WebSocket support, VAD, and pytest without changing existing model pins unnecessarily.
- [ ] Run protocol and server tests with fake providers; expect all to pass.

### Task 5: Add browser streaming client

**Files:**
- Create: `voice_agent/static/index.html`
- Create: `voice_agent/static/client.js`
- Create: `voice_agent/static/styles.css`
- Create: `tests/voice_agent/test_client_protocol.py`

- [ ] Define the browser state machine for disconnected, connecting, listening, thinking, speaking, and interrupted states.
- [ ] Implement microphone capture using MediaRecorder/AudioWorklet-compatible chunks, WebSocket JSON control messages, binary audio playback, and an explicit stop/mute control.
- [ ] On `speech_started` during playback, clear queued audio and send the interruption control event.
- [ ] Render transcript deltas, assistant text deltas, structured errors, and stage metrics without displaying credentials or raw reference paths.
- [ ] Run a browser smoke test against a fake WebSocket server; expect connect, stream, playback, interruption, and reconnect to pass.

### Task 6: Integrate consent flow, documentation, and benchmarks

**Files:**
- Modify: `app.py`
- Modify: `README.md`
- Create: `scripts/benchmark_voice_agent.py`
- Create: `tests/voice_agent/test_gradio_regression.py`

- [ ] Add a non-breaking Gradio action that registers a verified profile for the new service instead of exposing an unverified reference path.
- [ ] Add regression tests for the current sentence generation, transcription check, and XTTS wrapper behavior with mocked models.
- [ ] Document startup (`uvicorn voice_agent.server:app`), browser URL, local Ollama configuration, OpenAI-compatible environment variables, privacy defaults, and consent requirements.
- [ ] Implement the benchmark script to collect stage durations and print p50/p95 plus time-to-first-audio.
- [ ] Run the complete test suite and benchmark smoke run; record results without claiming a latency target that was not measured.

### Task 7: Final verification

**Files:**
- Modify only files required by failed checks.

- [ ] Run `pytest -q`.
- [ ] Run a local service startup check with mocked providers and verify the WebSocket handshake.
- [ ] Run the browser smoke test and confirm interruption prevents stale audio playback.
- [ ] Run `python scripts/benchmark_voice_agent.py --mock` and capture p50/p95 output.
- [ ] Inspect `git diff --check` and confirm no credentials, raw audio, or generated model artifacts are tracked.
- [ ] Attempt a focused commit per task; if repository permissions still block `.git` writes, leave the working tree ready for the user to commit locally.
