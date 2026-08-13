# Real-Time Voice Agent Design

**Date:** 2026-07-18  
**Status:** Design approved in conversation; implementation not started

## Goal

Evolve the existing local, consent-gated XTTS v2 voice-cloning project into a browser-based real-time conversational voice agent. The first vertical slice must support an open-ended assistant, while keeping provider boundaries clean enough to add multiple focused use cases later.

## Scope

The first release includes:

- A browser client using WebSocket streaming for microphone input and audio playback.
- A Python streaming service coordinating VAD, STT, LLM, TTS, interruption, and metrics.
- Mandatory consent verification before a voice profile can be registered or used.
- Pluggable provider interfaces for streaming STT, LLM, and TTS.
- Initial provider adapters for local Whisper/XTTS and Ollama, plus an OpenAI-compatible LLM adapter.
- Open-ended conversation configuration with system prompt, model/provider selection, language, and verified voice profile.
- Barge-in: new user speech cancels active LLM/TTS work and clears queued assistant audio.
- Per-stage latency, completion, cancellation, and error metrics.

Out of scope for this vertical slice: blockchain/NFT consent, production deepfake detection, invisible watermarking, mobile CoreML/ONNX deployment, Kubernetes, billing, and durable cloud audio storage.

## Architecture

The new implementation will live in a `voice_agent` package beside the existing `src/` modules.

### Consent and voice profiles

The existing consent flow remains available through the Gradio application while the new client is developed. A shared interface will create a signed local voice profile containing a voice ID, consent timestamp, language, and reference-audio metadata. The runtime must reject missing, expired, revoked, or mismatched profiles. Raw reference audio is not persisted by the runtime unless explicitly enabled.

### Provider interfaces

The service defines stable contracts for `STTProvider`, `LLMProvider`, and `TTSProvider`. Each contract exposes streaming operations, cancellation, and capability metadata. Provider selection is configuration-driven:

- Local STT: Whisper.
- Local TTS: XTTS v2.
- Local LLM: Ollama.
- Hosted LLM: OpenAI-compatible HTTP API.

The conversation layer must not depend on provider-specific SDK types.

### Conversation session

Each session owns the verified voice profile, provider instances, assistant configuration, bounded transcript history, current turn, cancellation state, and metrics context. Session state is isolated so concurrent browser sessions cannot share audio buffers or conversation history.

### WebSocket transport

The protocol uses JSON control/events plus binary audio frames. JSON events include `session_id`, `turn_id`, and timestamps. Initial event types are:

- `session_started`
- `speech_started`
- `speech_stopped`
- `transcript_delta`
- `assistant_text_delta`
- `response_audio`
- `turn_interrupted`
- `turn_completed`
- `metrics`
- `error`

The negotiated audio format is returned at session startup. Late chunks from cancelled turns are ignored using the turn ID.

## Data flow

1. The browser requests a session with a verified `voice_profile_id` and assistant configuration.
2. The server validates authorization, creates an isolated session, and returns the session ID and audio format.
3. The browser streams microphone frames over WebSocket.
4. Server-side VAD detects speech boundaries.
5. Streaming STT emits partial transcript deltas.
6. At end-of-turn, the conversation manager sends the user turn to the configured LLM.
7. LLM text is passed incrementally to TTS.
8. TTS audio chunks are sent to the browser for immediate playback.
9. If new speech is detected during playback, active LLM/TTS work is cancelled, queued audio is cleared, and a new turn begins.
10. The server emits final transcript, assistant text, completion, and latency metrics.

## Safety and failure handling

- Session startup fails closed for invalid voice authorization.
- A provider failure ends the affected turn where possible and returns a user-safe error event.
- TTS errors never fall back to an unverified voice.
- Disconnects cancel active model work and release buffers.
- Transcript history and audio buffers have explicit bounds.
- Credentials remain server-side and are loaded from environment/configuration.
- Audio is ephemeral by default; durable storage is opt-in and outside the MVP.

## Testing and acceptance criteria

### Tests

- Unit tests for consent/profile validation, provider contracts, session state, cancellation, and event schemas.
- Mock-provider integration tests for complete turns and barge-in.
- Local smoke tests for Ollama, Whisper, and XTTS where installed.
- Browser/WebSocket end-to-end tests for streaming, playback, reconnect, and interruption.
- Latency benchmarks reporting p50/p95 for VAD, STT, LLM, TTS, and total time-to-first-audio.

### Acceptance criteria

1. An unverified voice profile cannot start a session or synthesize speech.
2. A verified profile can complete an open-ended browser conversation.
3. Provider selection is configurable without changing conversation logic.
4. User interruption cancels active generation and prevents stale audio playback.
5. Disconnects and provider failures release resources and return structured errors.
6. Credentials and raw audio are not exposed or persisted by default.
7. Latency is measured by stage; sub-500 ms is an optimization target, not an initial guarantee with XTTS v2.

## Delivery sequence

1. Extract shared consent/profile interfaces without breaking the Gradio flow.
2. Add provider protocols and mock implementations.
3. Implement session state, cancellation, and event schemas.
4. Add the WebSocket service with local providers.
5. Add the browser streaming client.
6. Add Ollama and OpenAI-compatible LLM adapters.
7. Add integration tests, smoke tests, and latency reporting.
8. Add configurable assistant/use-case profiles after the open-ended slice is stable.
