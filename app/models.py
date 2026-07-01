"""
Request/response schema for the /chat endpoint.

Kept intentionally close to the assignment's exact spec -- the evaluator
does schema validation and "the schema is non-negotiable" per the brief.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1)


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str  # single-letter code, e.g. "K", "P" -- matches catalog convention


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False


class HealthResponse(BaseModel):
    status: str = "ok"
