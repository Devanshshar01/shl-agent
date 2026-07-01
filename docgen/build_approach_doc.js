const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow,
  TableCell, WidthType, ShadingType, BorderStyle, AlignmentType,
} = require("docx");

const PAGE_WIDTH_DXA = 12240;
const PAGE_HEIGHT_DXA = 15840;
const MARGIN = 720; // 0.5in

function h(text, level = HeadingLevel.HEADING_2) {
  return new Paragraph({
    heading: level,
    spacing: { before: 160, after: 60 },
    children: [new TextRun({ text, bold: true, color: "2E7D32" })],
  });
}

function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 80, ...opts.spacing },
    children: Array.isArray(text) ? text : [new TextRun({ text, ...opts.run })],
  });
}

function bullet(text, level = 0) {
  return new Paragraph({
    bullet: { level },
    spacing: { after: 40 },
    children: [new TextRun({ text, size: 20 })],
  });
}

function small(text, opts = {}) {
  return new TextRun({ text, size: 20, ...opts });
}

function boldSmall(text) {
  return new TextRun({ text, size: 20, bold: true });
}

const doc = new Document({
  sections: [
    {
      properties: {
        page: {
          size: { width: PAGE_WIDTH_DXA, height: PAGE_HEIGHT_DXA },
          margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN },
        },
      },
      children: [
        new Paragraph({
          spacing: { after: 40 },
          children: [
            new TextRun({ text: "Build a Conversational SHL Assessment Recommender", bold: true, size: 30, color: "2E7D32" }),
          ],
        }),
        new Paragraph({
          spacing: { after: 160 },
          children: [
            new TextRun({ text: "Approach Document — Devansh Sharma", size: 20, italics: true, color: "555555" }),
          ],
        }),

        h("Problem framing & scope"),
        p([small(
          "The task is a stateless, single-turn-at-a-time FastAPI service that turns a vague hiring need into a " +
          "grounded shortlist of SHL Individual Test Solutions through dialogue, while never inventing a URL and " +
          "staying inside an 8-message / 30-second-per-call budget. I treated grounding (never hallucinate) and " +
          "turn-budget discipline (clarify at most once) as the two non-negotiable constraints everything else " +
          "was designed around."
        )]),

        h("Architecture: retrieve-then-generate, one LLM call per turn"),
        p([small(
          "Each POST /chat call: (1) builds a retrieval query from the entire conversation history, not just the " +
          "latest message, since constraints accumulate turn over turn; (2) runs hybrid retrieval over the catalog " +
          "to get a bounded candidate pool (~18 items); (3) makes exactly one LLM call with a system prompt encoding " +
          "the four behaviors (clarify / recommend / refine / compare) plus the candidate pool and full history, " +
          "asking for strict JSON; (4) validates the model's JSON against the response schema and — critically — " +
          "cross-checks every recommended URL against the loaded catalog before it ever leaves the service, " +
          "silently dropping anything not found. This is the hard guarantee against hallucination; the system " +
          "prompt also instructs it, but the code does not trust that instruction alone."
        )]),
        p([small(
          "I deliberately avoided a multi-step agentic loop (retrieve → reason → retrieve again). A single grounded " +
          "call per turn is faster and easier to reason about under a 30s timeout, and one BM25 pass over a " +
          "keyword-dense catalog is normally sufficient. The trade-off: if retrieval misses badly on turn 1, the " +
          "agent cannot self-correct within that same turn — it has to rely on the next user turn to refine the query."
        )]),

        h("Retrieval: BM25 over embeddings"),
        p([small(
          "The catalog is a few hundred items with short, keyword-dense text — product names and skill terms like " +
          "\"SQL\", \"Docker\", \"OPQ32r\" dominate real recruiter queries far more than long free-text semantics. " +
          "BM25 (rank_bm25) over a synthetic search document (name + description + job levels + test-type labels) " +
          "handles this at least as well as embeddings, with zero external calls, zero cost, deterministic output, " +
          "and no index to host. I combine it with light structural filters (test-type, duration ceiling) and a " +
          "fallback: if a filter zeroes out every result, retrieval relaxes and returns the top unfiltered matches " +
          "rather than dead-ending the conversation with no candidates. This would not be the right call at " +
          "10,000+ items with long descriptions — that's the explicit trade-off, not an oversight."
        )]),

        h("Prompt design & turn-budget discipline"),
        p([small(
          "The system prompt encodes the four behaviors as explicit rules, forbids recommending anything outside " +
          "the injected CANDIDATE ITEMS block, and requires strict JSON with no markdown fencing. A concrete " +
          "constraint surfaced during testing against the ten provided traces: the evaluator counts turns as " +
          "user + assistant messages combined, capped at 8. A trace like C9 (7 real user turns of progressive " +
          "refinement) cannot complete if the agent asks two clarifying questions before committing. I tightened " +
          "the prompt to clarify at most once before producing an initial shortlist, then refine on later turns — " +
          "and the service itself forces a final, complete answer with end_of_conversation: true if a call arrives " +
          "at the cap, rather than risking a truncated or off-schema final turn."
        )]),

        h("Grounding & injection resistance"),
        new Paragraph({
          spacing: { after: 80 },
          children: [
            small("Two layers, not one: "),
            boldSmall("(1) prompt-level"),
            small(" — the system prompt instructs the model to only cite catalog items given to it, refuse general hiring/legal advice, and ignore in-message attempts to override its role; "),
            boldSmall("(2) code-level"),
            small(" — a regex pre-filter catches common injection patterns (\"ignore previous instructions\", \"reveal your system prompt\", etc.) before the LLM call even runs, and every returned URL is checked against the real catalog after the call, independent of what the model claims. Layer 2 is the one that actually matters for the hard-eval pass/fail; layer 1 is best-effort defense in depth."),
          ],
        }),

        h("Evaluation approach"),
        p([small(
          "I parsed the ten provided traces (C1–C10) into structured ground truth — each trace's sequence of user " +
          "turns plus its labeled final shortlist — and built a local replay harness (scripts/run_eval.py) that " +
          "plays the scripted user turns against a live /chat endpoint, checks hard-eval conditions (schema " +
          "compliance, catalog-only URLs, turn-cap honored) after every call, and computes Recall@10 against the " +
          "labeled shortlist at the end. This is a scripted replay, not SHL's LLM-simulated user, so it's a fast " +
          "local sanity check rather than a perfect predictor of the graded score — a simulated user paraphrases " +
          "and ad-libs more than a fixed script does. I also wrote unit tests isolating the grounding function " +
          "specifically: feeding it a mix of real and fabricated URLs and asserting only the real ones survive, " +
          "independent of any live model call, so that guarantee doesn't depend on the LLM behaving."
        )]),

        h("What didn't work / trade-offs"),
        bullet("Bare keyword BM25 initially under-ranked personality instruments (OPQ32r) for soft-skill phrasing like \"works with stakeholders\" when the catalog entry's description field was thin — mitigated by enriching the search document with job-level and test-type-label text, but this is the clearest place richer catalog descriptions (from the full scrape) will improve recall over the 35-item seed set."),
        bullet("An early version let the agent ask multiple clarifying questions across turns freely; replaying against C9 showed this reliably blew the 8-message cap on any trace with more than ~3 refinement turns. Fixed by budgeting clarification to at most one question turn in the prompt."),
        bullet("I considered a vector store (FAISS) for retrieval; skipped it after confirming on the seed catalog that BM25 alone hit correct top-5 matches for Excel/Word, safety, and technical-skill queries in unit tests — the added infra cost wasn't justified at this catalog size."),

        h("AI tool usage"),
        p([small(
          "Used Claude (Sonnet) as a coding assistant throughout — scaffolding the FastAPI structure, the BM25 " +
          "retrieval wrapper, the eval harness, and the trace-parsing regex — with each component reviewed, run, " +
          "and debugged against real test output rather than accepted as-is (e.g., the mock-server startup-handler " +
          "bug and the eval harness's turn-counting bug were both caught by actually running the tests and reading " +
          "the failures, not by inspection). Catalog scraping and grounding logic — the parts most load-bearing for " +
          "the hard-eval score — were written and verified with explicit unit tests rather than trusted blindly."
        )]),

        h("Stack"),
        p([small(
          "FastAPI + Pydantic for the API/schema; rank_bm25 for retrieval; Anthropic SDK for the LLM call (a " +
          "drop-in Gemini-compatible client is included for free-tier deployment); BeautifulSoup + requests for " +
          "the offline catalog scraper; Docker + render.yaml for one-command deployment to Render's free tier."
        )]),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  require("fs").writeFileSync("/home/claude/shl-agent/docgen/approach.docx", buf);
  console.log("wrote approach.docx");
});
