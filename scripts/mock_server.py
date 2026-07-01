"""
Runs the real FastAPI app but with Agent._client.messages.create monkey-
patched to a deterministic rule-based stub instead of a live Anthropic call.

This exists ONLY so we can smoke-test the full HTTP + retrieval + grounding
pipeline (including scripts/run_eval.py) in an environment with no API key
provisioned. It is NOT what gets deployed -- the real deployment uses a
real LLM call (see app/services/agent.py). Swap ANTHROPIC_API_KEY in and
run `uvicorn app.main:app` directly for the real thing.

The stub's "policy": after >=2 user turns worth of content, or if the
retrieved top candidate's search score looks confident, commit to a
shortlist made of the top retrieved candidates. Otherwise ask a generic
clarifying question. This is intentionally dumb -- it exists to exercise
the pipeline, not to be a good agent.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn

import app.main as main_mod


class FakeTextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class FakeResponse:
    def __init__(self, text):
        self.content = [FakeTextBlock(text)]


def make_stub(agent):
    real_search = agent.catalog.search

    def stub_create(*, model, max_tokens, temperature, system, messages):
        user_prompt = messages[0]["content"]
        # crude: count how many USER: lines are in the injected history block
        n_user_turns = user_prompt.count("USER:")
        candidates_block = user_prompt.split("CANDIDATE ITEMS")[1].split("CONVERSATION SO FAR")[0]
        candidate_urls = []
        for line in candidates_block.splitlines():
            if "https://" in line:
                url = line.split("|")[-1].strip()
                candidate_urls.append(url)

        force_close = "final allowed turn" in user_prompt

        if n_user_turns < 2 and not force_close:
            payload = {
                "reply": "Got it -- can you tell me a bit more about the seniority level or key skills?",
                "recommendations": [],
                "end_of_conversation": False,
            }
        else:
            picks = []
            for url in candidate_urls[:5]:
                item = agent.catalog.get_by_url(url)
                if item:
                    picks.append(item.to_recommendation())
            payload = {
                "reply": f"Here are {len(picks)} assessments that match what you've described.",
                "recommendations": picks,
                "end_of_conversation": True,
            }
        return FakeResponse(json.dumps(payload))

    return stub_create


if __name__ == "__main__":
    from app.services.catalog import Catalog
    from app.services.agent import Agent

    # The real app registers a startup handler that rebuilds _agent with a
    # live (unauthenticated, in this sandbox) Anthropic client. Clear it so
    # our stubbed agent below isn't clobbered when uvicorn fires startup
    # events.
    main_mod.app.router.on_startup.clear()

    main_mod._catalog = Catalog.load()
    main_mod._agent = Agent(main_mod._catalog)
    main_mod._agent._client.messages.create = make_stub(main_mod._agent)

    uvicorn.run(main_mod.app, host="0.0.0.0", port=8000, log_level="warning")
