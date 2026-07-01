# SHL Assessment Recommender

A conversational agent that helps hiring managers find SHL Individual Test
Solutions through dialogue, built for the SHL AI Intern take-home assignment.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in NVIDIA_API_KEY (preferred) or another provider key

# 1. Build the catalog (needs normal internet access -- not a
#    network-restricted sandbox):
cd scripts && python3 scrape_catalog.py --out ../app/data/catalog.json && cd ..

# 2. Run the service
uvicorn app.main:app --reload
```

Without step 1, the service falls back to `app/data/catalog.seed.json` --
a 35-item bootstrap set extracted verbatim from the 10 provided conversation
traces (see "Data" below). It's real data, just not the full catalog, so
you can develop against it but should NOT submit on it.

Check it's alive:

```bash
curl localhost:8000/health
curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hiring a Java developer who works with stakeholders"}]}'
```

## Project layout

```
app/
  main.py                 FastAPI app: GET /health, POST /chat
  models.py                Pydantic request/response schema
  services/
    catalog.py              Catalog loading + BM25 hybrid retrieval
    agent.py                 Core orchestration: retrieve -> one LLM call -> validate/ground
    llm_gemini.py             Optional Gemini-backed client (see "Swapping providers")
  data/
    catalog.seed.json        35 real items extracted from C1..C10 traces (dev/test fallback)
    catalog.json              (generated) full scrape -- not committed, build it yourself
scripts/
  scrape_catalog.py         Full catalog scraper (run with real internet access)
  extract_seed_catalog.py   Rebuilds catalog.seed.json from the C*.md traces
  parse_traces.py            Parses C1..C10 into tests/traces.json ground truth
  run_eval.py                 Local eval harness: replays traces, checks hard-evals + Recall@10
  mock_server.py            Runs the real app with a stubbed LLM call (no API key needed) -- dev/CI only
tests/
  test_catalog.py            Unit tests: retrieval quality, URL grounding
  test_agent_grounding.py    Unit tests: hallucination guard, injection detection, JSON parsing
  traces.json                (generated) parsed ground truth from the provided traces
Dockerfile
render.yaml                 One-click Render.com deploy config
```

## Design choices (see approach doc for the full writeup)

- **Retrieval: BM25, not embeddings.** The catalog is a few hundred
  keyword-dense product names/descriptions ("SQL", "Docker", "OPQ32r").
  BM25 handles this at least as well as embeddings, with zero external
  calls, zero cost, and deterministic results. Trade-off: would not scale
  to a much larger catalog with long free-text descriptions.
- **One LLM call per turn, not an agent loop.** Retrieval happens first
  (outside the LLM), then a single call gets the candidate pool + full
  history and returns strict JSON. Faster and easier to reason about than
  a multi-step ReAct loop, fits comfortably in the 30s/call budget.
- **Hard grounding, not prompt-only.** Every recommended URL is checked
  against the loaded catalog after the LLM responds; anything not in the
  catalog is silently dropped before the response leaves the service. The
  system prompt also instructs this, but the code doesn't trust it blindly.
- **Turn budget awareness.** The evaluator caps conversations at 8 messages
  total (user + assistant). The system prompt pushes the agent to clarify
  at most once before committing to an initial shortlist, and the service
  forces a final answer + `end_of_conversation: true` if a call arrives at
  the cap.

## Running the eval harness

```bash
# against a live server (needs ANTHROPIC_API_KEY set)
uvicorn app.main:app &
python3 scripts/run_eval.py --base-url http://localhost:8000
```

This replays each of the 10 provided traces' user turns against `/chat`,
checks hard-eval conditions (schema compliance, catalog-only URLs, turn cap),
and computes Recall@10 against each trace's labeled final shortlist. Results
go to `tests/eval_results.json`.

Note: this is a *scripted* replay (it plays back the trace's literal user
messages), not an LLM-simulated user like SHL's real evaluator. It's a
useful, fast local sanity check but won't perfectly predict the graded
score -- a simulated user will paraphrase and ad-lib more than a fixed script.

## Swapping the LLM provider

`app/services/agent.py` prefers `NVIDIA_API_KEY` when present, then
`GEMINI_API_KEY`, and finally `ANTHROPIC_API_KEY`. To force Gemini,
point the agent at `app/services/llm_gemini.py` directly:

```python
# in Agent.__init__:
from app.services.llm_gemini import GeminiClient
self._client = GeminiClient()  # reads GEMINI_API_KEY
```

The rest of the orchestration code is provider-agnostic; it only touches
`response.content[i].text`, which the Gemini shim reproduces.

## Deploying

**Render** (free tier): push this repo, connect it, and Render will pick up
`render.yaml` and `Dockerfile` automatically. Set `NVIDIA_API_KEY` (preferred)
or `GEMINI_API_KEY` / `ANTHROPIC_API_KEY` in the dashboard's environment
variables. Cold starts on the free tier can take up to ~1-2 minutes, which is
why `/health` in the spec explicitly allows that.

**Fly / Railway / Modal**: same `Dockerfile` works as-is; set the provider key
(`NVIDIA_API_KEY`, `GEMINI_API_KEY`, or `ANTHROPIC_API_KEY`) in each
platform's secrets UI.

Before submitting, make sure `app/data/catalog.json` (the full scrape) is
present in the image -- the Dockerfile copies everything under `app/`, so
just run `scripts/scrape_catalog.py` before building/pushing.

## Known limitations / what didn't fully fit

- The bundled `catalog.seed.json` (35 items from the traces) is NOT the
  full catalog -- run `scrape_catalog.py` before submitting.
- The scraper's CSS selectors are best-effort against the catalog site's
  structure as of this writing; if SHL changes the page markup, the
  selectors in `parse_listing_page`/`parse_detail_page` will need updating
  (check `scripts/.page_cache/` for cached HTML to debug against).
- Long conversations (5+ user turns) can hit the 8-message cap before a
  full refine-and-lock-in dialogue completes, if the agent asks more than
  one clarifying question. The prompt is tuned to minimize this but it's a
  real constraint of the format, not fully eliminable.
