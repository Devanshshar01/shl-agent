"""
Normalize a pasted SHL catalog export into the app's catalog schema.

The assignment requires using the full Individual Test Solutions catalog.
The pasted export is richer than the app's runtime schema and includes
report/profile artifacts that are not individual assessments, so this
script converts and filters it into app/data/catalog.json.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

KEY_TO_CODE = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}

MIN_URL_PREFIX = "https://www.shl.com/products/product-catalog/view/"


def _parse_duration_minutes(raw: str) -> int | None:
    if not raw:
        return None
    raw = raw.strip()
    if "untimed" in raw.lower():
        return None
    match = re.search(r"(\d+)", raw)
    return int(match.group(1)) if match else None


def _coerce_list_text(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(str(v).strip() for v in value if str(v).strip())
    if value is None:
        return ""
    return str(value).strip().strip(",")


def _normalize_test_types(keys: list[str]) -> tuple[list[str], list[str]]:
    codes = []
    labels = []
    for label in keys or []:
        code = KEY_TO_CODE.get(label)
        if code and code not in codes:
            codes.append(code)
            labels.append(label)
    return codes, labels


def should_keep(raw: dict) -> bool:
    name = str(raw.get("name", "")).strip()
    url = str(raw.get("link", "")).strip()
    if not name or not url.startswith(MIN_URL_PREFIX):
        return False
    return True


def convert_item(raw: dict) -> dict:
    name = str(raw.get("name", "")).strip()
    url = str(raw.get("link", "")).strip()
    test_type, test_type_labels = _normalize_test_types(raw.get("keys", []) or [])
    duration_raw = str(raw.get("duration_raw") or raw.get("duration") or "").strip()
    return {
        "id": str(raw.get("entity_id") or "").strip() or re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-"),
        "name": name,
        "url": url,
        "test_type": test_type,
        "test_type_labels": test_type_labels,
        "duration_minutes": _parse_duration_minutes(duration_raw),
        "duration_raw": duration_raw,
        "job_levels": _coerce_list_text(raw.get("job_levels")),
        "languages": _coerce_list_text(raw.get("languages")),
        "description": str(raw.get("description") or "").strip(),
        "source": "catalog_export",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to pasted export JSON")
    parser.add_argument("--out", default="../app/data/catalog.json")
    args = parser.parse_args()

    input_path = Path(args.input)
    raw_text = input_path.read_text(encoding="utf-8")
    raw_items = json.loads(raw_text, strict=False)

    converted = [convert_item(item) for item in raw_items if should_keep(item)]
    deduped = sorted({item["url"]: item for item in converted}.values(), key=lambda item: item["name"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(deduped, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(deduped)} catalog items to {out_path}")


if __name__ == "__main__":
    main()
