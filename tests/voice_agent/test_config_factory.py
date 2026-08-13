import pytest

from voice_agent.config import Settings
from voice_agent.factory import build_provider_factory


def test_default_settings_use_local_ollama(monkeypatch):
    for key in (
        "VOICE_AGENT_LLM_PROVIDER", "OLLAMA_BASE_URL", "OLLAMA_MODEL",
        "OPENAI_COMPATIBLE_BASE_URL", "OPENAI_COMPATIBLE_API_KEY", "OPENAI_COMPATIBLE_MODEL",
        "ASR_MODEL", "ASR_DEVICE", "VOICE_AGENT_DB",
    ):
        monkeypatch.delenv(key, raising=False)

    settings = Settings.from_env()

    assert settings.llm_provider == "ollama"
    assert settings.ollama_base_url == "http://127.0.0.1:11434"
    assert settings.asr_device == "auto"
    assert "OPENAI_COMPATIBLE_API_KEY" not in repr(settings)


def test_hosted_provider_requires_all_credentials(monkeypatch):
    monkeypatch.setenv("VOICE_AGENT_LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "https://example.test")
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_COMPATIBLE_MODEL", "model")

    with pytest.raises(ValueError, match="API key"):
        Settings.from_env()


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("VOICE_AGENT_LLM_PROVIDER", "unknown")

    with pytest.raises(ValueError, match="unsupported"):
        Settings.from_env()


def test_factory_constructs_without_network(monkeypatch, tmp_path):
    monkeypatch.setenv("VOICE_AGENT_DB", str(tmp_path / "profiles.sqlite3"))
    settings = Settings.from_env()

    factory = build_provider_factory(settings)
    stt, llm, tts = factory(type("Request", (), {})())

    assert stt.model_id == settings.asr_model
    assert llm.model == settings.ollama_model
    assert tts.model_id == "coqui/XTTS-v2"
