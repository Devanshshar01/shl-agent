"""
Optional drop-in replacement for the Anthropic client used in agent.py, for
running on Google's Gemini free tier instead (per the assignment's list of
suggested free LLM tiers). This mirrors just the subset of the Anthropic
Python SDK's `messages.create(...)` interface that agent.py relies on, so
swapping providers is a one-line change in Agent.__init__ rather than a
rewrite of the orchestration logic.

USAGE
-----
In app/services/agent.py, replace:

    self._client = anthropic.Anthropic()

with:

    from app.services.llm_gemini import GeminiClient
    self._client = GeminiClient()

...and set GEMINI_API_KEY instead of ANTHROPIC_API_KEY in your environment.
Everything else (system prompt, JSON parsing, grounding/validation) is
provider-agnostic and needs no changes, because agent.py only touches
`response.content[i].text` and `block.type == "text"`, both of which this
shim reproduces.

Not wired in by default because this sandbox ships the Anthropic SDK; swap
it in at deploy time if you'd rather run on Gemini's free tier.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)


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
        # Gemini's model names differ from Anthropic's; map a couple of
        # common aliases so callers can pass either without extra config.
        gemini_model = {
            "claude-sonnet-4-6": "gemini-2.5-flash",
        }.get(model, model)

        url = GEMINI_ENDPOINT.format(model=gemini_model)
        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [
                {"role": "user", "parts": [{"text": m["content"]}]} for m in messages
            ],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
                "responseMimeType": "application/json",
            },
        }
        resp = httpx.post(
            url,
            params={"key": self.api_key},
            json=body,
            timeout=25.0,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return _Response(content=[_TextBlock(type="text", text=text)])


class GeminiClient:
    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set")
        self.messages = _Messages(key)
