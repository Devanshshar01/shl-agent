import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.catalog import Catalog


def test_load():
    c = Catalog.load()
    assert len(c.items) > 0


def test_search_relevance_excel():
    c = Catalog.load()
    results = c.search("Excel Word admin assistant", top_k=5)
    names = [r.name for r in results]
    assert any("Excel" in n for n in names)
    assert any("Word" in n for n in names)


def test_search_relevance_safety():
    c = Catalog.load()
    results = c.search("chemical plant safety dependability", top_k=5)
    names = [r.name for r in results]
    assert any("Safety" in n or "Dependability" in n for n in names)


def test_search_never_returns_empty_for_nonsense():
    c = Catalog.load()
    # even a query with zero keyword overlap should fall back to *something*
    # rather than an empty list, so the agent always has candidates to work with
    results = c.search("zzz qqq nonexistent gibberish", top_k=5)
    assert len(results) > 0


def test_get_by_url_exact():
    c = Catalog.load()
    item = c.items[0]
    fetched = c.get_by_url(item.url)
    assert fetched is not None
    assert fetched.name == item.name


def test_get_by_url_hallucinated_returns_none():
    c = Catalog.load()
    fetched = c.get_by_url("https://www.shl.com/products/product-catalog/view/totally-made-up/")
    assert fetched is None


def test_test_type_filter():
    c = Catalog.load()
    results = c.search("assessment", top_k=50, test_types=["P"])
    assert all("P" in r.test_type for r in results)


if __name__ == "__main__":
    import traceback

    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError:
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
