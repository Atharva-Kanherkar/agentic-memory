from datetime import datetime, timezone
from dataclasses import dataclass

from models.base import MemoryRecord


@dataclass
class RankedResult:
    record: MemoryRecord
    raw_similarity: float
    recency_score: float
    importance_score: float
    final_score: float


def rank_results(
    results: list[tuple[MemoryRecord, float]],
    relevance_weight: float = 0.4,
    recency_weight: float = 0.3,
    importance_weight: float = 0.3,
    now: datetime | None = None,
) -> list[RankedResult]:
    """Re-rank (record, similarity) pairs using a weighted combination of
    relevance, recency, and importance. Returns highest-score first."""

    if not results:
        return []

    now = now or datetime.now(timezone.utc)

    # ── normalise recency to 0-1 ────────────────────────────────────────
    # Use last_accessed_at if set, otherwise created_at.
    # The most recent memory gets 1.0, the oldest gets 0.0.
    # If all timestamps are identical, everything gets 1.0.

    def _timestamp(record: MemoryRecord) -> datetime:
        return record.last_accessed_at or record.created_at

    ages = [(now - _timestamp(r)).total_seconds() for r, _ in results]
    max_age = max(ages)

    if max_age == 0:
        recency_scores = [1.0] * len(results)
    else:
        # age 0 → recency 1.0, max_age → recency 0.0
        recency_scores = [1.0 - (age / max_age) for age in ages]

    # ── compute final scores ────────────────────────────────────────────

    ranked = []
    for (record, similarity), recency in zip(results, recency_scores):
        final = (
            similarity * relevance_weight
            + recency * recency_weight
            + record.importance * importance_weight
        )
        ranked.append(RankedResult(
            record=record,
            raw_similarity=similarity,
            recency_score=recency,
            importance_score=record.importance,
            final_score=final,
        ))

    ranked.sort(key=lambda r: r.final_score, reverse=True)
    return ranked
