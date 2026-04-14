from types import SimpleNamespace

from apps.api.routers import discovery


def _book(goodreads_id: int) -> SimpleNamespace:
    return SimpleNamespace(id=goodreads_id, goodreads_id=goodreads_id)


def _request_with_state(
    *,
    bpr_path: str | None = None,
    metrics_path: str | None = None,
    qdrant: object | None = None,
) -> SimpleNamespace:
    state = SimpleNamespace(
        bpr_parquet_path=bpr_path,
        book_metrics_parquet_path=metrics_path,
        qdrant=qdrant,
    )
    return SimpleNamespace(app=SimpleNamespace(state=state))


def test_recommendations_blend_bpr_with_interaction_vector(monkeypatch):
    # BPR returns (id, score) pairs; scores descend so item 1 is most confident.
    bpr_scored = [(i, float(12 - i)) for i in range(1, 12)]
    monkeypatch.setattr(discovery, "_bpr_recommendations", lambda *_a, **_kw: bpr_scored)
    monkeypatch.setattr(discovery, "_cluster_vector_recommendations", lambda *_a, **_kw: [])
    monkeypatch.setattr(
        discovery,
        "_interaction_vector_recommendations",
        lambda *_a, **_kw: [101, 102, 103, 104],
    )
    # Fix bpr_weight so the assertion is deterministic (0.5 BPR / 0.5 vector, the new max).
    monkeypatch.setattr(discovery, "_bpr_weight", lambda *_a, **_kw: 0.5)
    monkeypatch.setattr(discovery, "_cold_start", lambda *_a, **_kw: [_book(i) for i in range(201, 230)])
    monkeypatch.setattr(
        discovery,
        "load_books_by_goodreads_ids",
        lambda _db, ids: [_book(int(i)) for i in ids],
    )
    monkeypatch.setattr(
        discovery,
        "serialize_books_with_engagement",
        lambda _db, books: [int(b.goodreads_id) for b in books],
    )

    current_user = SimpleNamespace(id=42, goodreads_id=999)
    request = _request_with_state(bpr_path="/tmp/bpr.parquet", qdrant=object())

    result = discovery.get_recommendations(limit=9, request=request, db=object(), current_user=current_user)

    # With bpr_weight=0.5, item 1 and vec-101 tie at 0.5 each; BPR items are
    # inserted first so item 1 wins the tie. Vector item 101 slots in at position 2.
    assert result == [1, 101, 2, 3, 4, 102, 5, 6, 7]


def test_recommendations_use_interactions_when_no_bpr_user(monkeypatch):
    # No goodreads_id → _bpr_recommendations is never called; bpr_scored stays [].
    monkeypatch.setattr(discovery, "_bpr_recommendations", lambda *_a, **_kw: [])
    monkeypatch.setattr(discovery, "_cluster_vector_recommendations", lambda *_a, **_kw: [])
    monkeypatch.setattr(
        discovery,
        "_interaction_vector_recommendations",
        lambda *_a, **_kw: [10, 11],
    )
    monkeypatch.setattr(discovery, "_cold_start", lambda *_a, **_kw: [_book(i) for i in [11, 12, 13, 14, 15]])
    monkeypatch.setattr(
        discovery,
        "load_books_by_goodreads_ids",
        lambda _db, ids: [_book(int(i)) for i in ids],
    )
    monkeypatch.setattr(
        discovery,
        "serialize_books_with_engagement",
        lambda _db, books: [int(b.goodreads_id) for b in books],
    )

    current_user = SimpleNamespace(id=7, goodreads_id=None)
    request = _request_with_state(bpr_path="/tmp/bpr.parquet", qdrant=object())

    result = discovery.get_recommendations(limit=5, request=request, db=object(), current_user=current_user)

    assert result == [10, 11, 12, 13, 14]
