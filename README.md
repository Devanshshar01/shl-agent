# SHL Assessment Recommender

A stateless FastAPI service that recommends SHL assessments through a
conversation-like API, built for the SHL AI intern take-home assignment.

---

## ✅ What it does

- Handles `POST /chat` with a full conversation history in `messages`.
- Returns schema-safe JSON with `reply`, `recommendations`, and
  `end_of_conversation`.
- Recommends 1–10 grounded catalog items once enough context exists.
- Returns an empty `recommendations` list when clarifying or refusing.
- Enforces catalog grounding: every returned URL is validated against the
  loaded catalog.
- Provides `GET /health` with `{"status": "ok"}`.

---

## 📦 Project layout

```text
app/
  main.py                 FastAPI app: GET /health, POST /chat
  models.py               Pydantic request/response schema
  services/
    catalog.py            Catalog loader + BM25-based retrieval
    agent.py              Conversation orchestration and grounding guard
    llm_gemini.py         Gemini-compatible provider wrapper
  data/
    catalog.seed.json     35-item fallback extracted from the provided traces
    catalog.json          Full generated catalog dataset (recommended for submit)
scripts/
  scrape_catalog.py       Catalog scraper for building app/data/catalog.json
  extract_seed_catalog.py Rebuilds catalog.seed.json from C*.md trace sources
  parse_traces.py         Parses trace source files into tests/traces.json
  run_eval.py             Local replay harness for evaluator-style checks
  mock_server.py          Dev stubbed LLM server (no API key needed)
tests/
  test_catalog.py         Catalog and retrieval unit tests
  test_agent_grounding.py Hallucination, injection, and output validation tests
  traces.json             Parsed conversation traces / ground truth
Dockerfile
render.yaml              Render deployment config
```

---

## 🔧 Requirements

- Python 3.12+
- Install dependencies with:

```bash
pip install -r requirements.txt
```

### Environment keys

- `NVIDIA_API_KEY` (preferred)
- `GEMINI_API_KEY`
- `ANTHROPIC_API_KEY`

The service prefers `NVIDIA_API_KEY`, then `GEMINI_API_KEY`, then
`ANTHROPIC_API_KEY`.

---

## 🚀 Setup and run

```bash
cp .env.example .env
# fill in NVIDIA_API_KEY or another provider key in .env

python scripts/scrape_catalog.py --out app/data/catalog.json
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

If you skip scraping, the service falls back to `app/data/catalog.seed.json`.
That fallback is fine for development, but the full catalog should be used
for final submission.

---

## 🧪 API contract

### `GET /health`

Response:

```json
{
  "status": "ok"
}
```

### `POST /chat`

Request:

```json
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "Okay, tell me more."},
    {"role": "user", "content": "Mid-level, around 4 years"}
  ]
}
```

Response:

```json
{
  "reply": "Got it. Here are grounded SHL assessments that fit your request.",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "OPQ32r", "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

- `recommendations` is empty during clarification or refusal.
- populated recommendations are between 1 and 10.
- `end_of_conversation` is `true` only when the agent is finished.

---

## 📈 Verification

### Smoke tests

I executed 10 random API smoke tests against the local service, covering:
- Java developer + stakeholder interaction
- personality assessment for customer service
- safety/compliance screening
- short SQL competency test
- graduate finance trainee
- OPQ32r vs GSA comparison
- vague "I need an assessment"
- Docker + AWS software engineer
- director leadership assessment
- inbound sales/customer service hiring

All 10 requests returned a valid schema and grounded catalog URLs.

### Local evaluator replay

Run the provided local harness:

```bash
python scripts/run_eval.py --base-url http://127.0.0.1:8000
```

This script replays the 10 supplied traces, verifies schema compliance, checks
that every recommendation URL is from the catalog, enforces the 8-message turn
cap, and computes Recall@10.

---

## 💡 Design summary

- **Stateless**: no per-conversation state is stored by the service.
- **Catalog grounding**: the code validates every recommended URL against the
  loaded catalog.
- **Behavior coverage**: clarify vague queries, recommend with enough context,
  refine on follow-up constraints, compare named items, and refuse off-topic
  requests.
- **Turn-budget aware**: the service is designed for the 8-message evaluator
  cap.
- **Balanced retrieval**: BM25 over the catalog avoids embedding cost and keeps
  recommendations deterministic and fast.

---

## ☁️ Deployment

The repository includes `Dockerfile` and `render.yaml` for deployment.
Ensure `app/data/catalog.json` is present in the container image by running
`python scripts/scrape_catalog.py --out app/data/catalog.json` before build.

Set the chosen provider key in the deployment environment.

---

## Notes

- The fallback `catalog.seed.json` is only for local dev and should not be
  the submitted runtime catalog.
- `scripts/run_eval.py` is a local playback check, not a perfect mirror of
  SHL's LLM-simulated evaluator.


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
