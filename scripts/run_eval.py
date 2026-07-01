"""
Local evaluation harness -- mirrors (a simplified version of) how SHL's
automated replay harness will grade the deployed service.

For each trace in tests/traces.json:
  - replay the trace's user turns IN ORDER as a real multi-turn conversation
    against POST /chat (scripted replay of the labeled user, not an LLM-
    simulated user -- good enough for local iteration; the real evaluator
    uses an LLM-simulated user which will paraphrase/ad-lib more)
  - after each call, run HARD EVAL checks (schema, turn cap, no
    out-of-catalog URLs)
  - after the final turn, compute Recall@10 against the trace's labeled
    final shortlist

Usage:
    python run_eval.py --base-url http://localhost:8000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

TRACES_PATH = Path(__file__).parent.parent / "tests" / "traces.json"
CATALOG_PATH = Path(__file__).parent.parent / "app" / "data"


def load_valid_urls() -> set[str]:
    for name in ("catalog.json", "catalog.seed.json"):
        p = CATALOG_PATH / name
        if p.exists():
            raw = json.loads(p.read_text(encoding="utf-8"))
            return {item["url"] for item in raw}
    return set()


def recall_at_10(predicted_urls: list[str], gold_urls: list[str]) -> float:
    if not gold_urls:
        return 1.0
    pred_set = set(predicted_urls[:10])
    gold_set = set(gold_urls)
    hit = len(pred_set & gold_set)
    return hit / len(gold_set)


def run_trace(client: httpx.Client, base_url: str, trace: dict, valid_urls: set[str]) -> dict:
    messages = []
    hard_eval_failures = []
    last_recommendations = []
    turn_count = 0

    for user_turn in trace["user_turns"]:
        messages.append({"role": "user", "content": user_turn})
        turn_count = len(messages)  # count messages (user+assistant), matching the spec's "8 turns including user & assistant"

        if turn_count > 8:
            hard_eval_failures.append(f"turn_cap_exceeded (turn {turn_count})")
            break

        try:
            resp = client.post(f"{base_url}/chat", json={"messages": messages}, timeout=30.0)
        except httpx.TimeoutException:
            hard_eval_failures.append(f"timeout_on_turn_{turn_count}")
            break

        if resp.status_code != 200:
            hard_eval_failures.append(f"http_{resp.status_code}_on_turn_{turn_count}")
            break

        try:
            data = resp.json()
        except Exception:
            hard_eval_failures.append(f"invalid_json_on_turn_{turn_count}")
            break

        for required_key in ("reply", "recommendations", "end_of_conversation"):
            if required_key not in data:
                hard_eval_failures.append(f"missing_field_{required_key}_turn_{turn_count}")

        recs = data.get("recommendations", []) or []
        if len(recs) > 10:
            hard_eval_failures.append(f"too_many_recommendations_turn_{turn_count}")

        for r in recs:
            if valid_urls and r.get("url") not in valid_urls:
                hard_eval_failures.append(f"out_of_catalog_url_turn_{turn_count}: {r.get('url')}")

        if recs:
            last_recommendations = [r["url"] for r in recs]

        messages.append({"role": "assistant", "content": data.get("reply", "")})

        if data.get("end_of_conversation"):
            break

    recall = recall_at_10(last_recommendations, trace["final_shortlist_urls"])

    return {
        "trace_id": trace["trace_id"],
        "hard_eval_failures": hard_eval_failures,
        "hard_eval_pass": len(hard_eval_failures) == 0,
        "recall_at_10": recall,
        "predicted": last_recommendations,
        "gold": trace["final_shortlist_urls"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    args = ap.parse_args()

    traces = json.loads(TRACES_PATH.read_text(encoding="utf-8"))
    valid_urls = load_valid_urls()

    # health check first
    with httpx.Client() as client:
        try:
            h = client.get(f"{args.base_url}/health", timeout=120.0)
            print(f"/health -> {h.status_code} {h.json()}", file=sys.stderr)
        except Exception as e:
            print(f"FATAL: /health check failed: {e}", file=sys.stderr)
            sys.exit(1)

        results = []
        for trace in traces:
            print(f"Running {trace['trace_id']}...", file=sys.stderr)
            t0 = time.time()
            result = run_trace(client, args.base_url, trace, valid_urls)
            result["elapsed_s"] = round(time.time() - t0, 2)
            results.append(result)
            status = "PASS" if result["hard_eval_pass"] else "FAIL"
            print(
                f"  {status}  recall@10={result['recall_at_10']:.2f}  "
                f"({result['elapsed_s']}s)  failures={result['hard_eval_failures']}",
                file=sys.stderr,
            )

    n = len(results)
    hard_pass_rate = sum(r["hard_eval_pass"] for r in results) / n
    mean_recall = sum(r["recall_at_10"] for r in results) / n

    print("\n=== SUMMARY ===")
    print(f"Traces run: {n}")
    print(f"Hard-eval pass rate: {hard_pass_rate:.0%}")
    print(f"Mean Recall@10: {mean_recall:.2f}")

    out_path = Path(__file__).parent.parent / "tests" / "eval_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nDetailed results -> {out_path}")


if __name__ == "__main__":
    main()
