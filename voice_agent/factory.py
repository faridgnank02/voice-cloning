"""Build provider and profile-store dependencies from validated settings."""

from __future__ import annotations

from voice_agent.config import Settings
from voice_agent.profiles import ProfileStore, VoiceProfile
from voice_agent.providers.local import LocalWhisperProvider, LocalXTTSProvider, OllamaProvider
from voice_agent.providers.openai_compatible import OpenAICompatibleProvider
from voice_agent.storage import SQLiteProfileRepository, SQLiteConsentVerifier


def build_profile_store(settings: Settings) -> ProfileStore:
    repository = SQLiteProfileRepository(settings.database_path)
    return ProfileStore(SQLiteConsentVerifier(repository), repository)


def build_provider_factory(settings: Settings):
    stt = LocalWhisperProvider(settings.asr_model, settings.asr_device)
    tts = LocalXTTSProvider()
    if settings.llm_provider == "ollama":
        llm = OllamaProvider(settings.ollama_base_url, settings.ollama_model)
    else:
        llm = OpenAICompatibleProvider(
            settings.openai_compatible_base_url,
            settings.openai_compatible_api_key,
            settings.openai_compatible_model,
        )

    def factory(request):
        return stt, llm, tts

    return factory
