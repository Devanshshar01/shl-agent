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
                raw = json.loads(p.read_text())
                if not raw:
                    continue
                return cls._from_raw(raw)

        raise FileNotFoundError(
            f"No catalog file found in {DATA_DIR}. Run scripts/scrape_catalog.py "
            "or scripts/extract_seed_catalog.py first."
        )

    @classmethod
    def _from_raw(cls, raw: list[dict]) -> "Catalog":
        items = []
        for r in raw:
            search_doc = " ".join(
                filter(
                    None,
                    [
                        r.get("name", ""),
                        r.get("description", ""),
                        r.get("job_levels", ""),
                        " ".join(r.get("test_type_labels", []) or []),
                        r.get("id", "").replace("-", " "),
                    ],
                )
            )
            items.append(
                CatalogItem(
                    id=r.get("id", ""),
                    name=r["name"],
                    url=r["url"],
                    test_type=r.get("test_type", []) or [],
                    duration_minutes=r.get("duration_minutes"),
                    languages=r.get("languages", ""),
                    job_levels=r.get("job_levels", ""),
                    description=r.get("description", ""),
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
