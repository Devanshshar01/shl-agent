"""
Loads the scraped SHL catalog and provides retrieval over it.

RETRIEVAL DESIGN
-----------------
We use BM25 (rank_bm25) over a synthetic "search document" per catalog item
(name + description + job_levels + test_type labels), combined with light
structural boosts (test-type filter, duration ceiling, remote/adaptive
flags) rather than a vector DB.

Why not embeddings? The catalog is a few hundred items with short, mostly
keyword-dense text (product names, skill names like "SQL", "Docker", "OPQ").
BM25 handles exact/near-exact keyword matches (which dominate this domain --
recruiters say "Java", "Excel", "safety") at least as well as embeddings,
with zero external API calls, zero cost, deterministic results, and no cold
embedding index to build/host. This is a defensible trade-off for a catalog
this size; it would NOT be the right call at 50k+ items with long free-text
descriptions, and the approach doc says so explicitly.

The retriever returns candidates; the LLM never sees the raw catalog file,
only the top-N candidates for the current turn. This is the grounding
mechanism that prevents URL hallucination: the model can only mention items
that were actually retrieved and passed into its context this turn.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from rank_bm25 import BM25Okapi

DATA_DIR = Path(__file__).parent.parent / "data"

TEST_TYPE_LEGEND = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgment",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}

_TOKEN_RE = re.compile(r"[a-z0-9+#.]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class CatalogItem:
    id: str
    name: str
    url: str
    test_type: list[str]
    duration_minutes: int | None
    languages: str
    job_levels: str
    description: str
    search_doc: str = field(default="", repr=False)

    @property
    def test_type_str(self) -> str:
        return "".join(self.test_type) if self.test_type else ""

    @property
    def test_type_labels(self) -> list[str]:
        return [TEST_TYPE_LEGEND.get(c, c) for c in self.test_type]

    def to_recommendation(self) -> dict:
        return {
            "name": self.name,
            "url": self.url,
            "test_type": self.test_type_str or "P",
        }

    def to_prompt_line(self) -> str:
        dur = f"{self.duration_minutes} min" if self.duration_minutes is not None else "duration unspecified"
        labels = ", ".join(self.test_type_labels) or "unspecified type"
        jl = f" | levels: {self.job_levels}" if self.job_levels else ""
        return f"- {self.name} | {dur} | {labels}{jl} | {self.url}"


class Catalog:
    def __init__(self, items: list[CatalogItem]):
        self.items = items
        self.by_url = {i.url: i for i in items}
        self.by_name_lower = {i.name.lower(): i for i in items}
        corpus = [_tokenize(i.search_doc) for i in items]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Catalog":
        """
        Loads the richest available catalog file. Preference order:
        1. catalog.json      -- full scrape via scripts/scrape_catalog.py
        2. catalog.seed.json -- bootstrap set extracted from provided traces
        """
        if path:
            candidates = [Path(path)]
        else:
            candidates = [DATA_DIR / "catalog.json", DATA_DIR / "catalog.seed.json"]

        for p in candidates:
            if p.exists():
                raw = json.loads(p.read_text(encoding="utf-8"))
                if not raw:
                    continue
                return cls._from_raw(raw)

        raise FileNotFoundError(
            f"No catalog file found in {DATA_DIR}. Run scripts/scrape_catalog.py "
            "or scripts/extract_seed_catalog.py first."
        )

    @staticmethod
    def _normalize_records(raw: object) -> list[dict]:
        if isinstance(raw, list):
            return [r for r in raw if isinstance(r, dict)]
        if isinstance(raw, dict):
            for key in ("products", "items", "catalog", "data", "results"):
                value = raw.get(key)
                if isinstance(value, list):
                    return [r for r in value if isinstance(r, dict)]
            if any(key in raw for key in ("entity_id", "title", "link", "name", "url")):
                return [raw]
        return []

    @staticmethod
    def _coerce_test_type(raw: dict) -> list[str]:
        raw_value = raw.get("test_type")
        if isinstance(raw_value, str):
            raw_value = [raw_value]
        elif raw_value is None:
            raw_value = []
        elif not isinstance(raw_value, list):
            raw_value = [str(raw_value)]

        if raw_value:
            normalized: list[str] = []
            for value in raw_value:
                if isinstance(value, str):
                    value = value.strip()
                    if not value:
                        continue
                    code = value.upper()
                    if len(code) == 1 and code in TEST_TYPE_LEGEND:
                        normalized.append(code)
                    else:
                        normalized.append(TEST_TYPE_LEGEND.get(value, value))
            if normalized:
                return normalized

        labels = raw.get("test_type_labels") or raw.get("testTypeLabels") or []
        if isinstance(labels, str):
            labels = [labels]
        elif not isinstance(labels, list):
            labels = [str(labels)]

        normalized_labels = []
        for label in labels:
            if not isinstance(label, str):
                continue
            label = label.strip()
            if not label:
                continue
            normalized_labels.append(label)

        if normalized_labels:
            reverse_legend = {v: k for k, v in TEST_TYPE_LEGEND.items()}
            inferred = []
            for label in normalized_labels:
                code = reverse_legend.get(label)
                if code:
                    inferred.append(code)
                elif len(label) == 1 and label.upper() in TEST_TYPE_LEGEND:
                    inferred.append(label.upper())
                else:
                    inferred.append(label)
            return inferred

        return []

    @staticmethod
    def _coerce_duration(raw: dict) -> int | None:
        for key in ("duration_minutes", "duration", "estimated_duration_minutes"):
            value = raw.get(key)
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)
            if isinstance(value, str):
                digits = re.search(r"(\d+)", value)
                if digits:
                    return int(digits.group(1))
        return None

    @classmethod
    def _from_raw(cls, raw: object) -> "Catalog":
        items = []
        for r in cls._normalize_records(raw):
            name = r.get("name") or r.get("title") or r.get("product_name") or r.get("product") or ""
            url = r.get("url") or r.get("link") or r.get("href") or r.get("product_url") or ""
            if not name or not url:
                continue

            search_doc = " ".join(
                filter(
                    None,
                    [
                        name,
                        r.get("description", "") or r.get("summary", "") or r.get("details", ""),
                        r.get("job_levels", "") or r.get("jobLevels", "") or r.get("levels", ""),
                        " ".join(r.get("test_type_labels", []) or r.get("testTypeLabels", []) or []),
                        str(r.get("id") or r.get("entity_id") or r.get("entityId") or r.get("product_id") or "").replace("-", " "),
                    ],
                )
            )
            items.append(
                CatalogItem(
                    id=str(r.get("id") or r.get("entity_id") or r.get("entityId") or r.get("product_id") or ""),
                    name=name,
                    url=url,
                    test_type=cls._coerce_test_type(r),
                    duration_minutes=cls._coerce_duration(r),
                    languages=r.get("languages", "") or r.get("language", ""),
                    job_levels=r.get("job_levels", "") or r.get("jobLevels", "") or r.get("levels", ""),
                    description=r.get("description", "") or r.get("summary", "") or r.get("details", ""),
                    search_doc=search_doc,
                )
            )
        return cls(items)

    def search(
        self,
        query: str,
        top_k: int = 15,
        test_types: list[str] | None = None,
        max_duration: int | None = None,
    ) -> list[CatalogItem]:
        """Hybrid retrieval: BM25 ranking, then optional structural filters."""
        if not self._bm25 or not self.items:
            return []

        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens) if tokens else [0.0] * len(self.items)
        ranked = sorted(zip(self.items, scores), key=lambda p: p[1], reverse=True)

        results = []
        for item, score in ranked:
            if test_types and not (set(item.test_type) & set(test_types)):
                continue
            if max_duration is not None and item.duration_minutes is not None:
                if item.duration_minutes > max_duration:
                    continue
            results.append(item)
            if len(results) >= top_k:
                break

        # Fallback: if filters zeroed everything out, relax and return
        # unfiltered top matches rather than returning nothing (better to
        # let the LLM see near-misses and explain a trade-off than to
        # dead-end the conversation).
        if not results:
            results = [item for item, _ in ranked[:top_k]]

        return results

    def get_by_name(self, name: str) -> CatalogItem | None:
        return self.by_name_lower.get(name.strip().lower())

    def get_by_url(self, url: str) -> CatalogItem | None:
        return self.by_url.get(url.strip())

    def all_names(self) -> list[str]:
        return [i.name for i in self.items]
