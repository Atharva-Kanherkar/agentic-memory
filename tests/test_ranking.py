"""Verify that the ranking function respects weight tuning.

Scenario: two records with identical content and identical similarity scores,
but different timestamps. Changing the recency weight should flip the ranking.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone, timedelta
from models.semantic import SemanticMemory
from retrieval.ranking import rank_results


def make_record(content: str, created_at: datetime, importance: float = 0.5) -> SemanticMemory:
    return SemanticMemory(content=content, created_at=created_at, importance=importance)


def test_recency_weight():
    now = datetime.now(timezone.utc)
    old = make_record("Python was created by Guido", created_at=now - timedelta(days=30))
    new = make_record("Python was created by Guido", created_at=now - timedelta(minutes=5))

    # Identical similarity scores — only recency and importance differ
    results = [(old, 0.90), (new, 0.90)]

    # ── high recency weight: newer should win ───────────────────────────
    ranked = rank_results(results, relevance_weight=0.2, recency_weight=0.6, importance_weight=0.2, now=now)
    assert ranked[0].record is new, (
        f"Expected newer record first, got: {ranked[0].record.created_at}"
    )
    print(f"  PASS  high recency weight → newer wins")
    print(f"         new: {ranked[0].final_score:.4f}  old: {ranked[1].final_score:.4f}")

    # ── zero recency weight: should tie (same similarity + importance) ──
    ranked = rank_results(results, relevance_weight=0.5, recency_weight=0.0, importance_weight=0.5, now=now)
    assert abs(ranked[0].final_score - ranked[1].final_score) < 1e-9, (
        "Expected tied scores when recency_weight=0"
    )
    print(f"  PASS  zero recency weight → tied scores")
    print(f"         both: {ranked[0].final_score:.4f}")


def test_importance_weight():
    now = datetime.now(timezone.utc)
    low = make_record("Some fact", created_at=now, importance=0.2)
    high = make_record("Some fact", created_at=now, importance=0.9)

    results = [(low, 0.85), (high, 0.85)]

    ranked = rank_results(results, relevance_weight=0.2, recency_weight=0.2, importance_weight=0.6, now=now)
    assert ranked[0].record is high, "Expected high-importance record first"
    print(f"  PASS  high importance weight → important record wins")
    print(f"         high: {ranked[0].final_score:.4f}  low: {ranked[1].final_score:.4f}")


def test_relevance_dominates():
    now = datetime.now(timezone.utc)
    weak = make_record("Weak match", created_at=now, importance=1.0)
    strong = make_record("Strong match", created_at=now - timedelta(days=365), importance=0.1)

    results = [(weak, 0.50), (strong, 0.95)]

    ranked = rank_results(results, relevance_weight=0.7, recency_weight=0.15, importance_weight=0.15, now=now)
    assert ranked[0].record is strong, "Expected high-relevance record first despite low importance and old age"
    print(f"  PASS  high relevance weight → best match wins despite being old and low importance")
    print(f"         strong: {ranked[0].final_score:.4f}  weak: {ranked[1].final_score:.4f}")


def test_empty():
    ranked = rank_results([])
    assert ranked == []
    print(f"  PASS  empty input → empty output")


if __name__ == "__main__":
    print("Ranking tests:\n")
    test_recency_weight()
    test_importance_weight()
    test_relevance_dominates()
    test_empty()
    print("\nAll tests passed.")
