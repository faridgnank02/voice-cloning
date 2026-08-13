"""Environment-backed configuration for the runnable voice-agent service."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_path: str
    llm_provider: str
    ollama_base_url: str
    ollama_model: str
    openai_compatible_base_url: str
    openai_compatible_api_key: str
    openai_compatible_model: str
    asr_model: str
    asr_device: str

    @classmethod
    def from_env(cls) -> "Settings":
        provider = os.getenv("VOICE_AGENT_LLM_PROVIDER", "ollama").strip().lower()
        if provider not in {"ollama", "openai-compatible"}:
            raise ValueError(f"unsupported LLM provider: {provider}")
        values = cls(
            database_path=os.getenv("VOICE_AGENT_DB", "data/voice_profiles.sqlite3"),
            llm_provider=provider,
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "llama3:8b"),
            openai_compatible_base_url=os.getenv("OPENAI_COMPATIBLE_BASE_URL", ""),
            openai_compatible_api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY", ""),
            openai_compatible_model=os.getenv("OPENAI_COMPATIBLE_MODEL", ""),
            asr_model=os.getenv("ASR_MODEL", "openai/whisper-base"),
            asr_device=os.getenv("ASR_DEVICE", "auto"),
        )
        if provider == "ollama" and (not values.ollama_base_url or not values.ollama_model):
            raise ValueError("Ollama base URL and model are required")
        if provider == "openai-compatible":
            if not values.openai_compatible_base_url:
                raise ValueError("OpenAI-compatible base URL is required")
            if not values.openai_compatible_api_key:
                raise ValueError("OpenAI-compatible API key is required")
            if not values.openai_compatible_model:
                raise ValueError("OpenAI-compatible model is required")
        return values

    def __repr__(self) -> str:
        return (
            "Settings(database_path={!r}, llm_provider={!r}, ollama_base_url={!r}, "
            "ollama_model={!r}, openai_compatible_base_url={!r}, "
            "openai_compatible_api_key='<redacted>', openai_compatible_model={!r}, "
            "asr_model={!r}, asr_device={!r})"
        ).format(
            self.database_path, self.llm_provider, self.ollama_base_url,
            self.ollama_model, self.openai_compatible_base_url,
            self.openai_compatible_model, self.asr_model, self.asr_device,
        )
