"""
Pulls every markdown-table product row out of the 10 provided conversation
traces (C1..C10) into a de-duplicated seed catalog.

This is NOT a substitute for the full scrape_catalog.py run against the live
site -- it's a ~20-30 item, 100%-real-data bootstrap so the FastAPI service,
retrieval, and eval harness can be built and tested end-to-end today. Every
row here came from the assignment's own labeled traces, so names/urls/types
are trustworthy; what's missing is full catalog *coverage* and the
description/job_levels detail-page fields (filled with best-effort blanks).
"""

import json
import re
from pathlib import Path

TRACE_DIR = Path(__file__).parent / "traces_source"
OUT_PATH = Path(__file__).parent.parent / "app" / "data" / "catalog.seed.json"

ROW_RE = re.compile(
    r"\|\s*\d+\s*\|\s*(?P<name>[^|]+?)\s*\|\s*(?P<test_type>[^|]+?)\s*\|\s*(?P<keys>[^|]+?)\s*\|\s*"
    r"(?P<duration>[^|]+?)\s*\|\s*(?P<languages>[^|]+?)\s*\|\s*<(?P<url>https?://[^>]+)>\s*\|"
)


def normalize_duration(raw: str):
    raw = raw.strip()
    if raw in ("—", "-", "", "Untimed"):
        return None if raw != "Untimed" else 0
    m = re.search(r"(\d+)", raw)
    return int(m.group(1)) if m else None


def main():
    catalog = {}
    for md_file in sorted(TRACE_DIR.glob("C*.md")):
        text = md_file.read_text()
        for m in ROW_RE.finditer(text):
            name = m.group("name").strip()
            url = m.group("url").strip()
            test_type = [c for c in m.group("test_type").strip() if c.isalpha()]
            keys = m.group("keys").strip()
            duration_raw = m.group("duration").strip()
            languages = m.group("languages").strip()

            if url in catalog:
                # merge: keep richer language/duration info if this dup has more
                continue

            catalog[url] = {
                "id": re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-"),
                "name": name,
                "url": url,
                "test_type": test_type,
                "test_type_labels": keys.split(", ") if keys else [],
                "duration_minutes": normalize_duration(duration_raw),
                "duration_raw": duration_raw,
                "languages": languages,
                "job_levels": "",
                "description": "",
                "source": md_file.name,
            }

    entries = sorted(catalog.values(), key=lambda e: e["name"])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    print(f"Extracted {len(entries)} unique catalog entries -> {OUT_PATH}")
    for e in entries:
        print(f"  - {e['name']}  [{''.join(e['test_type'])}]  {e['url']}")


if __name__ == "__main__":
    main()
