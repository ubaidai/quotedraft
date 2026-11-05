from quotedraft.store import rrf
from quotedraft.text import normalize, tokenize


def test_size_notation_normalises_the_same():
    a, b, c = tokenize("M12x50 hex bolt"), tokenize("M12 x 50 hex bolt"), tokenize("m12 X 50mm Hex Bolt")
    assert a == b
    assert set(a) <= set(c)


def test_inch_fractions_become_decimals():
    assert "1.5 inch" in normalize('1-1/2" pipe')
    assert "1.5 inch" in normalize("1-1/2 inch pipe")
    assert "0.5 inch" in normalize('Ball valve DN15 (1/2")')


def test_trade_terms_are_expanded_not_replaced():
    t = tokenize("Hex hd bolt 20mm dia BZP")
    assert "m20" in t and "bzp" in t and "zinc" in t


def test_rrf_weights_prefer_weighted_list():
    fused = rrf([["a", "b"], ["b", "a"]], weights=[1.0, 5.0])
    assert fused[0][0] == "b"


def test_exact_catalog_name_is_top_hit(store, catalog):
    for p in catalog[::17]:
        assert store.search(p.name, k=1, mode="keyword")[0].sku == p.sku


def test_hybrid_returns_k_real_products(store):
    res = store.search("DN25 stainless ball valve flanged", k=5)
    assert len(res) == 5
    assert all(store.get(c.sku) for c in res)
