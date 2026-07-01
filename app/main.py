from __future__ import annotations

import logging
import time

from dotenv import load_dotenv

load_dotenv()  # must run before Agent() reads any API key env vars

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.models import ChatRequest, ChatResponse, HealthResponse
from app.services.agent import Agent
from app.services.catalog import Catalog

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shl-agent")

app = FastAPI(title="SHL Assessment Recommender", version="0.1.0")

_catalog: Catalog | None = None
_agent: Agent | None = None


def _ensure_agent() -> Agent:
    global _catalog, _agent
    if _catalog is None:
        _catalog = Catalog.load()
    if _agent is None:
        _agent = Agent(_catalog)
    return _agent


@app.get("/debug/provider")
def debug_provider() -> dict[str, str]:
    agent = _ensure_agent()
    client = agent._client
    provider = type(client).__name__ if client is not None else "offline"
    return {"provider": provider}


@app.on_event("startup")
def _startup() -> None:
    global _catalog, _agent
    _catalog = Catalog.load()
    _agent = Agent(_catalog)
    logger.info("Loaded catalog with %d items", len(_catalog.items))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request) -> ChatResponse:
    start = time.monotonic()
    try:
        result = _ensure_agent().handle(req.messages)
    except Exception:
        logger.exception("agent.handle failed")
        result = ChatResponse(
            reply="Something went wrong on my end -- could you try again?",
            recommendations=[],
            end_of_conversation=False,
        )
    elapsed = time.monotonic() - start
    logger.info("chat turn handled in %.2fs (turns=%d)", elapsed, len(req.messages))
    return result


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=200,  # keep schema-shaped even on failure, per evaluator's hard-eval req
        content=ChatResponse(
            reply="Something went wrong on my end -- could you try again?",
            recommendations=[],
            end_of_conversation=False,
        ).model_dump(),
    )