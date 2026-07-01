"""
Scrapes the SHL Individual Test Solutions catalog into a single structured
JSON file the agent can load at startup.

WHY THIS EXISTS
----------------
The take-home spec requires the agent to be grounded ONLY in the real SHL
catalog (https://www.shl.com/solutions/products/product-catalog/), scoped to
Individual Test Solutions (Pre-packaged Job Solutions are explicitly out of
scope). We can't hot-link the live site at chat time (latency, availability,
and it would break the "stateless, <30s per call" requirement), so we scrape
once, offline, and ship a static catalog.json that the FastAPI service loads
into memory + a vector index at startup.

USAGE
-----
    python scrape_catalog.py --out ../app/data/catalog.json

Run this from an environment with normal internet access (your laptop, CI,
or the deploy target's build step) -- not from a sandboxed dev container
with an egress allowlist.

DESIGN NOTES
------------
- The catalog site paginates a "Pre-packaged Job Solutions" table and an
  "Individual Test Solutions" table separately on the same listing pages.
  We only keep rows from the Individual Test Solutions table.
- Each row links to a detail page with duration, languages, description,
  and the "Test Type" key legend (A/B/C/D/E/K/P/S...). We follow every
  link and scrape the detail page too, because the listing page alone is
  missing description/duration for most rows in the traces we were given.
- Being polite: small delay between requests, retry with backoff, a real
  User-Agent, and we cache each fetched detail page to disk so a re-run
  after a crash doesn't refetch pages we already have.
"""

import argparse
import json
import re
import time
import sys
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE = "https://www.shl.com"
LISTING_URL = BASE + "/products/product-catalog/"
CACHE_DIR = Path(__file__).parent / ".page_cache"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

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
                time.sleep(0.6)  # be polite
                return resp
            last_exc = RuntimeError(f"HTTP {resp.status_code} for {url}")
        except requests.RequestException as e:
            last_exc = e
        time.sleep(1.5 * (attempt + 1))
    raise last_exc


def discover_listing_pages() -> list[str]:
    """
    The catalog listing is paginated. We start at page 0 and keep going
    (both for the unified list and, if the site splits tables by type/
    start letter, we detect and follow "next" links) until no new
    Individual Test Solutions rows are found.
    """
    urls = []
    start = 0
    empty_streak = 0
    while empty_streak < 2:
        page_url = LISTING_URL
        params = {"start": start, "type": "1"}  # type=1 => Individual Test Solutions, per SHL's own filter param
        resp = _get(page_url, params=params)
        soup = BeautifulSoup(resp.content, "html.parser")
        rows = soup.select("table tr") or soup.select("[class*='product']")
        if not rows or len(rows) <= 1:
            empty_streak += 1
        else:
            empty_streak = 0
            urls.append(page_url + f"?start={start}&type=1")
        start += 12
        if start > 2000:  # sane upper bound guard
            break
    return urls


def parse_listing_page(html: bytes) -> list[dict]:
    """Extract row-level fields available directly on the listing table."""
    soup = BeautifulSoup(html, "html.parser")
    items = []
    for row in soup.select("tr[data-course-id], tr.product-row, table tr"):
        link_tag = row.find("a", href=True)
        if not link_tag:
            continue
        href = link_tag["href"]
        if "/product-catalog/view/" not in href:
            continue
        name = link_tag.get_text(strip=True)
        remote_testing = bool(row.select_one("[class*='remote']"))
        adaptive_irt = bool(row.select_one("[class*='adaptive']"))
        key_cell = row.select_one("[class*='key']")
        keys = key_cell.get_text(strip=True) if key_cell else ""
        items.append(
            {
                "name": name,
                "url": urljoin(BASE, href),
                "remote_testing": remote_testing,
                "adaptive_irt": adaptive_irt,
                "test_type_raw": keys,
            }
        )
    return items


def parse_detail_page(html: bytes) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    def text_after_label(label: str) -> str | None:
        node = soup.find(string=re.compile(label, re.I))
        if not node:
            return None
        parent = node.find_parent()
        if not parent:
            return None
        sib = parent.find_next_sibling()
        return sib.get_text(strip=True) if sib else None

    description_tag = soup.select_one("[class*='description']") or soup.find("p")
    description = description_tag.get_text(strip=True) if description_tag else ""

    duration = text_after_label("Assessment length") or text_after_label("Duration")
    job_levels = text_after_label("Job level")
    languages_tag = soup.select_one("[class*='language']")
    languages = languages_tag.get_text(strip=True) if languages_tag else ""

    return {
        "description": description,
        "duration_raw": duration or "",
        "job_levels": job_levels or "",
        "languages_raw": languages,
    }


def normalize_duration(raw: str) -> int | None:
    if not raw:
        return None
    m = re.search(r"(\d+)", raw)
    return int(m.group(1)) if m else None


def normalize_test_type(raw: str) -> list[str]:
    return [c for c in raw.strip() if c in TEST_TYPE_LEGEND]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../app/data/catalog.json")
    ap.add_argument("--limit", type=int, default=None, help="cap items, for testing")
    args = ap.parse_args()

    print("Discovering listing pages...", file=sys.stderr)
    pages = discover_listing_pages()
    print(f"Found {len(pages)} listing pages", file=sys.stderr)

    all_items: dict[str, dict] = {}
    for page_url in pages:
        try:
            resp = _get(page_url)
        except Exception as e:
            print(f"  skip {page_url}: {e}", file=sys.stderr)
            continue
        for item in parse_listing_page(resp.content):
            all_items[item["url"]] = item

    print(f"Found {len(all_items)} unique Individual Test Solutions", file=sys.stderr)

    urls = list(all_items.keys())
    if args.limit:
        urls = urls[: args.limit]

    catalog = []
    for i, url in enumerate(urls):
        try:
            resp = _get(url)
            detail = parse_detail_page(resp.content)
        except Exception as e:
            print(f"  detail fetch failed for {url}: {e}", file=sys.stderr)
            detail = {"description": "", "duration_raw": "", "job_levels": "", "languages_raw": ""}

        base = all_items[url]
        entry = {
            "id": re.sub(r"[^a-z0-9]+", "-", base["name"].lower()).strip("-"),
            "name": base["name"],
            "url": base["url"],
            "test_type": normalize_test_type(base["test_type_raw"]),
            "remote_testing": base["remote_testing"],
            "adaptive_irt": base["adaptive_irt"],
            "duration_minutes": normalize_duration(detail["duration_raw"]),
            "duration_raw": detail["duration_raw"],
            "job_levels": detail["job_levels"],
            "languages": detail["languages_raw"],
            "description": detail["description"],
        }
        catalog.append(entry)
        if (i + 1) % 25 == 0:
            print(f"  scraped {i + 1}/{len(urls)}", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False))
    print(f"Wrote {len(catalog)} entries to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
