"""OpenAI-compatible streaming chat-completion adapter."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping, Sequence

import httpx

from voice_agent.providers.base import ProviderError


class OpenAICompatibleProvider:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.model = model
        self.timeout = timeout

    def __repr__(self) -> str:
        return (
            f"OpenAICompatibleProvider(base_url={self.base_url!r}, "
            f"model={self.model!r}, timeout={self.timeout!r})"
        )

    async def stream_response(
        self,
        messages: Sequence[Mapping[str, str]],
        system_prompt: str,
    ) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "stream": True,
            "messages": [{"role": "system", "content": system_prompt}, *messages],
        }
        async for line in self._stream_lines(payload=payload):
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                item = json.loads(data)
                content = item["choices"][0]["delta"].get("content", "")
            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
                raise ValueError("malformed OpenAI-compatible response") from error
            if content:
                yield content

    async def _stream_lines(self, *, payload: dict) -> AsyncIterator[str]:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST", f"{self.base_url}/v1/chat/completions", json=payload, headers=headers
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        yield line
        except httpx.HTTPError as error:
            raise ProviderError("OpenAI-compatible request failed") from error
