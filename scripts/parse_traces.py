"""
Parses the provided C1..C10 conversation traces into a structured JSON file:
for each trace, the sequence of user turns (as ground-truth user replies the
eval harness can play back) and the FINAL shortlist of catalog URLs the
labeled agent converged on (used as the Recall@10 ground truth).

This intentionally does NOT try to replay every intermediate agent turn --
per the assignment, the real evaluator simulates a user from a persona/fact
sheet and lets a live LLM conversation unfold naturally against our /chat
endpoint. What we CAN reuse deterministically from these traces is:
  (a) the sequence of user messages, which make a reasonable scripted replay
      for local smoke-testing, and
  (b) the final labeled shortlist, which is a genuine Recall@10 target.
"""

import json
import re
from pathlib import Path

TRACE_DIR = Path(__file__).parent / "traces_source"
OUT_PATH = Path(__file__).parent.parent / "tests" / "traces.json"

USER_RE = re.compile(r"\*\*User\*\*\s*\n\s*\n>\s*(.+?)(?=\n\n\*\*Agent\*\*)", re.DOTALL)
URL_RE = re.compile(r"<(https?://[^>]+)>")
END_RE = re.compile(r"`end_of_conversation`:\s*\*\*(true|false)\*\*")


def parse_trace(path: Path) -> dict:
    text = path.read_text()

    # user turns: grab the quoted block after each **User** header
    user_turns = []
    for m in USER_RE.finditer(text):
        raw = m.group(1)
        # strip leading '> ' continuation markers from multi-line quotes
        lines = [re.sub(r"^>\s?", "", ln) for ln in raw.split("\n")]
        user_turns.append("\n".join(lines).strip())

    # also catch a final trailing user turn with no following **Agent** (rare)
    all_user_blocks = re.findall(r"\*\*User\*\*\s*\n\s*\n((?:>.*\n?)+)", text)
    if len(all_user_blocks) > len(user_turns):
        raw = all_user_blocks[-1]
        lines = [re.sub(r"^>\s?", "", ln) for ln in raw.split("\n")]
        user_turns.append("\n".join(lines).strip())

    # final shortlist: URLs in the LAST markdown table before the final
    # end_of_conversation: true (or just the last table in the file)
    end_matches = list(END_RE.finditer(text))
    is_complete = any(m.group(1) == "true" for m in end_matches)

    # take the last occurring block of table rows in the document
    all_urls_in_order = URL_RE.findall(text)
    # dedupe preserving order, take from the LAST table specifically:
    # split on '### Turn' and inspect the last turn block that has a table
    turn_blocks = re.split(r"### Turn \d+", text)
    final_urls = []
    for block in reversed(turn_blocks):
        urls = URL_RE.findall(block)
        if urls:
            seen = set()
            for u in urls:
                if u not in seen:
                    final_urls.append(u)
                    seen.add(u)
            break

    return {
        "trace_id": path.stem,
        "user_turns": user_turns,
        "final_shortlist_urls": final_urls,
        "labeled_complete": is_complete,
    }


def main():
    traces = []
    for md_file in sorted(TRACE_DIR.glob("C*.md"), key=lambda p: int(re.search(r"\d+", p.stem).group())):
        traces.append(parse_trace(md_file))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(traces, indent=2, ensure_ascii=False))
    print(f"Parsed {len(traces)} traces -> {OUT_PATH}")
    for t in traces:
        print(f"  {t['trace_id']}: {len(t['user_turns'])} user turns, {len(t['final_shortlist_urls'])} final urls")


if __name__ == "__main__":
    main()
