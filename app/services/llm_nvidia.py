"""
Optional drop-in replacement for the Anthropic client used in agent.py,
for running on NVIDIA's OpenAI-compatible API.

This mirrors just the subset of the Anthropic SDK interface that agent.py
relies on, so swapping providers is a one-line change in Agent.__init__.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

NVIDIA_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"


@dataclass
class _TextBlock:
    type: str
    text: str


@dataclass
class _Response:
    content: list[_TextBlock]


class _Messages:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        temperature: float,
        system: str,
        messages: list[dict],
    ) -> _Response:
        nvidia_model = {
            "claude-sonnet-4-6": "meta/llama-3.1-70b-instruct",
        }.get(model, model)

        payload_messages = [{"role": "system", "content": system}]
        payload_messages.extend(messages)

        body = {
            "model": nvidia_model,
            "messages": payload_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        resp = httpx.post(
            NVIDIA_ENDPOINT,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=body,
            timeout=25.0,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return _Response(content=[_TextBlock(type="text", text=text)])


class NvidiaClient:
    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("NVIDIA_API_KEY")
        if not key:
            raise RuntimeError("NVIDIA_API_KEY not set")
        self.messages = _Messages(key)