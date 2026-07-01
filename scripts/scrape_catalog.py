"""
Build the SHL catalog JSON from public SHL pages.

The product-catalog route is JS-heavy, but SHL exposes enough of the catalog
through the public site map and search results pages to build a grounded local
catalog for the agent. We use those pages as discovery seeds, then scrape the
individual product pages for title, description, duration, language info, and
other metadata.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

BASE = "https://www.shl.com"
SITE_MAP_URL = BASE + "/site-map/"
SEARCH_URL = BASE + "/search/"
CACHE_DIR = Path(__file__).parent / ".page_cache"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

DISCOVERY_QUERIES = [
    "assessment",
    "assessments",
    "personality",
    "behavioral",
    "cognitive",
    "skills",
    "simulation",
    "coding",
    "language",
    "technical",
    "call center",
    "business",
    "manager",
    "sales",
    "graduate",
    "aptitude",
    "ability",
    "opq",
    "mq",
    "gsa",
    "sjt",
    "rjp",
    "verify",
    "360",
    "interview",
    "psychometric",
    "verbal",
    "numerical",
    "abstract",
    "mechanical",
    "deductive",
    "inductive",
    "administrative",
    "customer service",
    "leadership",
    "motivation",
    "situational",
    "problem solving",
    "reasoning",
    "work style",
    "communication",
    "attention",
    "accuracy",
    "customer",
    "retail",
    "manufacturing",
    "bpo",
    "sales hiring",
    "tech hiring",
    "manager development",
    "succession planning",
    "talent mobility",
    "video",
    "feedback",
    "development center",
    "assessment center",
]

DETAIL_URL_SKIP = {
    "/products/",
    "/products/360/",
    "/products/assessments/",
    "/products/assessments/assessment-and-development-centers/",
    "/products/assessments/behavioral-assessments/",
    "/products/assessments/cognitive-assessments/",
    "/products/assessments/job-focused-assessments/",
    "/products/assessments/personality-assessment/",
    "/products/assessments/skills-and-simulations/",
    "/products/video-interviews/",
    "/products/video-feedback/",
    "/products/product-catalog/",
    "/products/assessments/aptitude-tests/",
    "/products/assessments/psychometric-tests/",
}

TEST_TYPE_HINTS = [
    ("/products/360/", ["D"]),
    ("/products/assessments/assessment-and-development-centers/", ["E"]),
    ("/products/assessments/personality-assessment/", ["P"]),
    ("/products/assessments/behavioral-assessments/", ["B"]),
    ("/products/assessments/cognitive-assessments/", ["A"]),
    ("/products/assessments/skills-and-simulations/call-center-simulations/", ["S"]),
    ("/products/assessments/skills-and-simulations/coding-simulations/", ["S"]),
    ("/products/assessments/skills-and-simulations/business-skills/", ["K"]),
    ("/products/assessments/skills-and-simulations/technical-skills/", ["K"]),
    ("/products/assessments/skills-and-simulations/language-evaluation/", ["K"]),
    ("/products/assessments/job-focused-assessments/", ["A"]),
    ("/products/video-interviews/", ["S"]),
    ("/products/video-feedback/", ["S"]),
]


def _get(url: str, params: dict | None = None, retries: int = 3) -> requests.Response:
    CACHE_DIR.mkdir(exist_ok=True)
    cache_key = re.sub(r"[^a-zA-Z0-9]+", "_", url + json.dumps(params or {}))[:180]
    cache_file = CACHE_DIR / f"{cache_key}.html"
    if cache_file.exists():
        resp = requests.models.Response()
        resp._content = cache_file.read_bytes()
        resp.status_code = 200
        return resp

    last_exc = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=20)
            if resp.status_code == 200:
                cache_file.write_bytes(resp.content)
                time.sleep(0.4)
                return resp
            last_exc = RuntimeError(f"HTTP {resp.status_code} for {url}")
        except requests.RequestException as exc:
            last_exc = exc
        time.sleep(1.0 * (attempt + 1))
    raise last_exc


def _normalise_url(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/") + "/"


def _is_product_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.path.startswith("/products/") and parsed.path not in DETAIL_URL_SKIP


def _discover_urls_from_html(html: bytes) -> set[str]:
    soup = BeautifulSoup(html, "html.parser")
    urls = set()
    for anchor in soup.find_all("a", href=True):
        href = urljoin(BASE, anchor["href"])
        if _is_product_url(href):
            urls.add(_normalise_url(href))
    return urls


def discover_product_urls() -> list[str]:
    discovered = set()

    site_map_html = _get(SITE_MAP_URL).content
    discovered.update(_discover_urls_from_html(site_map_html))

    for query in DISCOVERY_QUERIES:
        search_html = _get(SEARCH_URL, params={"q": query}).content
        discovered.update(_discover_urls_from_html(search_html))

    frontier = sorted(discovered)
    for _ in range(2):
        next_frontier = []
        for url in frontier:
            try:
                html = _get(url).content
            except Exception:
                continue
            for child in _discover_urls_from_html(html):
                if child not in discovered:
                    discovered.add(child)
                    next_frontier.append(child)
        frontier = next_frontier
        if not frontier:
            break

    return sorted(discovered)


def _first_match(patterns: list[str], text: str) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip() if match.groups() else match.group(0).strip()
    return ""


def _infer_test_type(url: str, title: str, text: str) -> list[str]:
    path = urlparse(url).path.lower()
    for hint, value in TEST_TYPE_HINTS:
        if hint in path:
            return value

    haystack = f"{title} {text}".lower()
    if any(token in haystack for token in ["opq", "personality", "motivation"]):
        return ["P"]
    if any(token in haystack for token in ["gsa", "sjt", "rjp", "ucf", "behavior"]):
        return ["B"]
    if any(token in haystack for token in ["verify", "aptitude", "ability", "reasoning", "numerical", "verbal", "abstract", "mechanical"]):
        return ["A"]
    if any(token in haystack for token in ["simulation", "interview", "video"]):
        return ["S"]
    if "360" in haystack:
        return ["D"]
    return ["K"]


def parse_detail_page(html: bytes, url: str) -> dict | None:
    soup = BeautifulSoup(html, "html.parser")

    def meta(attr: str, value: str) -> str:
        tag = soup.find("meta", attrs={attr: value})
        return tag.get("content", "").strip() if tag else ""

    title = meta("property", "og:title") or (soup.title.get_text(strip=True) if soup.title else "")
    if not title or title in {"Our Products | SHL", "Search Results | SHL"}:
        return None

    main = soup.find("main") or soup.body or soup
    text = " ".join(main.stripped_strings)
    description = meta("name", "description")
    if not description:
        paragraphs = [p.get_text(" ", strip=True) for p in main.find_all("p") if p.get_text(strip=True)]
        description = next((p for p in paragraphs if len(p) > 40), title)

    duration_raw = _first_match(
        [
            r"\b(\d{1,3})\s+minutes?\b",
            r"\b(?:takes|take|lasts|last)\s+(\d{1,3})\s+minutes?\b",
            r"\b(\d{1,3})\s+min\b",
        ],
        text,
    )
    if duration_raw:
        duration_raw = f"{duration_raw} minutes"

    languages_raw = _first_match(
        [
            r"in any of\s+(\d{1,3}\s+languages?)",
            r"in\s+(\d{1,3}\s+languages?)",
            r"available in\s+(\d{1,3}\s+languages?)",
        ],
        text,
    )

    job_levels = _first_match([r"job levels?[:\s]+([^.|\n]+)"], text)
    if not (duration_raw or languages_raw or job_levels or description != title):
        return None

    return {
        "title": title,
        "description": description,
        "duration_raw": duration_raw,
        "job_levels": job_levels,
        "languages_raw": languages_raw,
        "body_text": text,
        "test_type": _infer_test_type(url, title, text),
    }


def normalize_duration(raw: str) -> int | None:
    if not raw:
        return None
    m = re.search(r"(\d+)", raw)
    return int(m.group(1)) if m else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="../app/data/catalog.json")
    parser.add_argument("--limit", type=int, default=None, help="cap items, for testing")
    args = parser.parse_args()

    print("Discovering product URLs...", file=sys.stderr)
    urls = discover_product_urls()
    print(f"Found {len(urls)} candidate product URLs", file=sys.stderr)
    if args.limit:
        urls = urls[: args.limit]

    catalog = []
    for index, url in enumerate(urls, start=1):
        try:
            html = _get(url).content
            detail = parse_detail_page(html, url)
        except Exception as exc:
            print(f"  detail fetch failed for {url}: {exc}", file=sys.stderr)
            continue

        if not detail:
            continue

        title = detail["title"]
        catalog.append(
            {
                "id": re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-"),
                "name": title,
                "url": url,
                "test_type": detail["test_type"],
                "remote_testing": False,
                "adaptive_irt": False,
                "duration_minutes": normalize_duration(detail["duration_raw"]),
                "duration_raw": detail["duration_raw"],
                "job_levels": detail["job_levels"],
                "languages": detail["languages_raw"],
                "description": detail["description"],
            }
        )

        if index % 25 == 0:
            print(f"  scraped {index}/{len(urls)}", file=sys.stderr)

    seed_path = Path(__file__).parent.parent / "app" / "data" / "catalog.seed.json"
    if seed_path.exists():
        try:
            seed_items = json.loads(seed_path.read_text())
            catalog = list(seed_items) + catalog
        except Exception:
            pass

    catalog = sorted({item["url"]: item for item in catalog}.values(), key=lambda e: e["name"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False))
    print(f"Wrote {len(catalog)} entries to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
