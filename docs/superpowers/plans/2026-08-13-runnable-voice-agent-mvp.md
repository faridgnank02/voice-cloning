# Runnable Voice Agent MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the existing consent-backed voice-agent prototype into a runnable local MVP where the Gradio consent flow creates a persistent verified profile and a configured browser WebSocket service can conduct and interrupt conversations.

**Architecture:** Keep the existing `voice_agent` package as the runtime boundary. Replace its in-memory-only profile wiring with a small local SQLite profile store, add an explicit environment-driven provider factory, and connect Gradio registration to that store. Then improve the WebSocket path with audio normalization, server-side turn detection, incremental event delivery, browser end-to-end coverage, and documented startup commands.

**Tech Stack:** Python 3.10+, SQLite via the standard library, FastAPI/Uvicorn, WebSocket, httpx, existing Whisper/XTTS integrations, Ollama/OpenAI-compatible HTTP APIs, pytest/pytest-asyncio, and Playwright when available.

## Global Constraints

- Mandatory consent verification before a voice profile can be registered or used.
- TTS must never fall back to an unverified voice.
- Credentials remain server-side and are never serialized into browser events.
- Raw audio is ephemeral by default; only reference-audio paths needed for local synthesis are stored.
- Provider selection must be configurable without changing conversation orchestration.
- Transcript history and audio buffers have explicit bounds.
- Existing Gradio consent and cloning behavior must continue working.
- The MVP does not include blockchain/NFT consent, watermarking, deepfake detection, mobile CoreML/ONNX deployment, billing, or Kubernetes.
- Do not claim sub-500 ms latency until a model-backed benchmark measures it.

---

## File map

- Create `voice_agent/storage.py`: SQLite-backed profile persistence, schema creation, revocation, and consent lookup.
- Modify `voice_agent/profiles.py`: allow `ProfileStore` to use a persistent backend while retaining strict verifier checks.
- Create `voice_agent/config.py`: typed environment configuration and safe defaults.
- Create `voice_agent/factory.py`: provider construction from configuration, with local and hosted LLM selection.
- Modify `voice_agent/server.py`: use configured profile store/factory, serve the browser client, and stream events incrementally.
- Create `voice_agent/audio.py`: browser-audio normalization and bounded frame/turn assembly.
- Modify `voice_agent/session.py`: consume normalized turns and emit cancellable event streams instead of requiring post-turn event flushing.
- Modify `app.py`: register verified profiles after the existing consent check and expose profile ID/status without exposing raw audio paths.
- Modify `voice_agent/static/client.js`: use server turn events, display states, and robust playback interruption.
- Create `tests/voice_agent/test_storage.py`: persistence/revocation/restart tests.
- Create `tests/voice_agent/test_config_factory.py`: provider/configuration tests with no network calls.
- Create `tests/voice_agent/test_audio.py`: format validation, bounds, turn detection, and malformed input tests.
- Modify `tests/voice_agent/test_server.py` and `tests/voice_agent/test_session.py`: incremental streaming and reconnect/interruption coverage.
- Create `tests/voice_agent/test_browser_protocol.py`: browser protocol fixture tests; use Playwright only when installed.
- Modify `requirements.txt`: add only the dependencies required by the implementation.
- Modify `README.md`: exact setup, profile registration, Ollama/OpenAI-compatible configuration, and manual test procedure.
- Modify `scripts/benchmark_voice_agent.py`: report stage timings for mock and model-backed runs.
- Create `Dockerfile` and `docker-compose.yml`: optional service packaging without bundling model weights or secrets.

## Task 1: Persistent consent-backed profile store

**Files:**
- Create: `voice_agent/storage.py`
- Modify: `voice_agent/profiles.py`
- Modify: `app.py`
- Create: `tests/voice_agent/test_storage.py`
- Modify: `tests/voice_agent/test_profiles.py`

**Interfaces:**
- `SQLiteProfileRepository(database_path: str)` with `initialize()`, `save(profile)`, `get(profile_id)`, `revoke(profile_id)`, and `close()`.
- `ProfileStore(consent_verifier, repository=None)` continues exposing `register(profile)`, `get(profile_id)`, and `assert_usable(profile_id, language)`.
- `register_verified_profile(reference_audio_path, consent_id, language) -> VoiceProfile` is the Gradio-facing function; it must call the existing consent result only after `SentenceMatcher.passed` is true.

- [ ] Write failing tests proving a registered profile survives a new `ProfileStore` instance, revoked profiles remain unusable after restart, and raw audio bytes are not copied into SQLite.
- [ ] Run `pytest tests/voice_agent/test_storage.py tests/voice_agent/test_profiles.py -v`; expect collection/attribute failures for the repository interface.
- [ ] Implement a parameterized SQLite schema storing profile ID, consent ID, aware consent timestamp, language, reference path, and revocation state; use parameterized SQL and create the parent directory safely.
- [ ] Implement repository reads/writes and make `ProfileStore` delegate persistence when a repository is supplied while preserving strict registration/use-time consent verification.
- [ ] Add the Gradio registration action only on the successful consent branch; return the opaque profile ID and a user-facing status string, never the filesystem path.
- [ ] Run the focused tests and then `pytest -q`; expected result is all existing tests plus the new persistence tests passing.

## Task 2: Configured provider factory and runnable service

**Files:**
- Create: `voice_agent/config.py`
- Create: `voice_agent/factory.py`
- Modify: `voice_agent/server.py`
- Create: `tests/voice_agent/test_config_factory.py`
- Modify: `requirements.txt`

**Interfaces:**
- `Settings.from_env() -> Settings` reads `VOICE_AGENT_DB`, `VOICE_AGENT_LLM_PROVIDER`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OPENAI_COMPATIBLE_BASE_URL`, `OPENAI_COMPATIBLE_API_KEY`, `OPENAI_COMPATIBLE_MODEL`, `ASR_MODEL`, and `ASR_DEVICE`.
- `build_profile_store(settings) -> ProfileStore` creates the SQLite repository and a verifier that validates signed consent records.
- `build_provider_factory(settings) -> ProviderFactory` returns local Whisper/XTTS plus the selected Ollama or OpenAI-compatible LLM without network calls during construction.
- `app = create_configured_app()` is the production entrypoint; tests continue using dependency injection through `create_app(...)`.

- [ ] Write failing tests for default local configuration, missing hosted credentials, unknown provider names, and secret redaction from `repr(settings)`.
- [ ] Run the focused config tests; expect failures because configuration/factory modules do not exist.
- [ ] Implement typed settings with explicit validation: Ollama requires a base URL/model; OpenAI-compatible mode requires base URL, API key, and model; no secret is included in logs or browser payloads.
- [ ] Wire `voice_agent.server:app` to the configured app while retaining test injection.
- [ ] Add a startup test that imports the app with a temporary database and mock providers without downloading models.
- [ ] Run `pytest tests/voice_agent/test_config_factory.py tests/voice_agent/test_server.py -v` and then `pytest -q`.

## Task 3: Audio normalization, bounded turns, and server-side turn detection

**Files:**
- Create: `voice_agent/audio.py`
- Modify: `voice_agent/server.py`
- Modify: `voice_agent/session.py`
- Create: `tests/voice_agent/test_audio.py`
- Modify: `tests/voice_agent/test_server.py`

**Interfaces:**
- `AudioFormat(sample_rate: int, channels: int, sample_width: int, encoding: str)`.
- `AudioTurnAssembler(format, max_bytes, silence_ms)`, with `push(frame) -> list[AudioEvent]`, `flush() -> bytes`, and `reset()`.
- `decode_browser_chunk(chunk: bytes, content_type: str) -> PCMFrame` rejects unknown/oversized formats instead of passing arbitrary WebM blobs to Whisper.

- [ ] Write failing tests for accepted PCM/WAV frames, rejected oversized or malformed chunks, speech-start/speech-stop events, and max-turn enforcement.
- [ ] Run `pytest tests/voice_agent/test_audio.py -v`; expect missing-module failures.
- [ ] Implement deterministic framing and a lightweight energy-based VAD for local MVP use; keep VAD behind the interface so WebRTC VAD can replace it later.
- [ ] Update WebSocket handling to emit `speech_started`/`speech_stopped`, feed only normalized PCM to STT, and reject invalid frames with structured errors.
- [ ] Run audio/server tests and the full suite.

## Task 4: Incremental session event delivery and interruption

**Files:**
- Modify: `voice_agent/session.py`
- Modify: `voice_agent/server.py`
- Modify: `voice_agent/static/client.js`
- Modify: `tests/voice_agent/test_session.py`
- Modify: `tests/voice_agent/test_server.py`

**Interfaces:**
- `ConversationSession.events()` is an async iterator of `SessionEvent` values.
- `ConversationSession.interrupt(turn_id: int | None = None)` cancels only the active turn and prevents stale text/audio events.
- The WebSocket endpoint forwards each event immediately with `session_id` and `turn_id`; it does not wait for `finish_turn` to flush a list.

- [ ] Write failing tests proving transcript deltas, assistant text deltas, audio chunks, completion, and metrics arrive before the turn completes.
- [ ] Write a failing barge-in test proving an interrupt cancels the active LLM/TTS task and no later audio from that turn reaches the client.
- [ ] Implement an event queue per session with bounded capacity and turn-aware cancellation.
- [ ] Update the WebSocket loop to multiplex incoming audio/control messages with outgoing session events using asyncio tasks and clean shutdown in `finally`.
- [ ] Update the browser client to render listening/thinking/speaking/interrupted states and clear queued audio on interruption.
- [ ] Run session/server tests and `pytest -q`.

## Task 5: Browser end-to-end and manual smoke testing

**Files:**
- Create: `tests/voice_agent/test_browser_protocol.py`
- Modify: `voice_agent/static/index.html`
- Modify: `voice_agent/static/client.js`
- Modify: `README.md`

**Interfaces:**
- Browser sends `start_session`, binary audio frames, `finish_turn`, and `interrupt`.
- Browser handles `session_started`, `speech_started`, `speech_stopped`, `transcript_delta`, `assistant_text_delta`, `response_audio`, `turn_interrupted`, `turn_completed`, `metrics`, and `error`.

- [ ] Add protocol-fixture tests that validate the browser event reducer without requiring a microphone or model.
- [ ] Add a Playwright smoke test when Playwright is installed: load `/`, connect with a test profile, send fixture audio through a fake provider app, verify transcript/audio display, interrupt playback, and reconnect.
- [ ] Add a documented manual test using Chrome/Safari microphone permissions and a temporary verified profile.
- [ ] Run protocol tests; run Playwright tests when available and report skipped status explicitly when it is not installed.

## Task 6: Packaging, observability, and final verification

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Modify: `scripts/benchmark_voice_agent.py`
- Modify: `README.md`
- Modify: `requirements.txt`
- Create: `tests/voice_agent/test_packaging.py`

- [ ] Write failing tests for settings-driven health output, benchmark stage fields, and Docker configuration that does not copy secrets/model artifacts.
- [ ] Implement `/healthz`, structured stage timing for VAD/STT/LLM/TTS/time-to-first-audio, graceful WebSocket shutdown, and bounded log fields.
- [ ] Add a non-root Docker image and compose service with environment-variable configuration, healthcheck, and a writable data volume only for SQLite metadata.
- [ ] Document exact commands for local mock mode, Ollama mode, OpenAI-compatible mode, profile registration, and model-backed latency measurement.
- [ ] Run `pytest -q`, `python3 scripts/benchmark_voice_agent.py --mock --iterations 20`, `python3 -m compileall -q voice_agent scripts`, and `git diff --check`.
- [ ] Inspect tracked-file candidates for API keys, raw audio, model weights, and generated databases; none may be included.

## Explicitly deferred after this plan

The following remain separate projects after the runnable MVP: consent blockchain/NFT issuance, invisible audio watermarking, production deepfake detection, mobile CoreML/ONNX conversion, multi-tenant authentication/billing, Kubernetes deployment, and a true sub-500 ms model-optimized benchmark.
