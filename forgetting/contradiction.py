from __future__ import annotations

from dataclasses import dataclass

from models.semantic import SemanticMemory
from stores.semantic_store import SemanticStore


@dataclass(frozen=True, slots=True)
class ContradictionCandidate:
    record: SemanticMemory
    similarity: float


class ContradictionDetector:
    """Retrieve contradiction candidates and persist confirmed supersession links."""

    def find_potential_contradictions(
        self,
        new_record: SemanticMemory,
        store: SemanticStore,
        threshold: float = 0.85,
        top_k: int = 5,
    ) -> list[ContradictionCandidate]:
        stored_record = store.get_by_id(new_record.id)
        if stored_record is None:
            raise ValueError(f"Semantic record '{new_record.id}' must be stored before contradiction lookup")
        if stored_record.embedding is None:
            raise ValueError(f"Semantic record '{new_record.id}' is missing an embedding")

        raw_results = store.retrieve_by_vector(stored_record.embedding, top_k=max(1, top_k + 1))
        candidates = [
            ContradictionCandidate(record=record, similarity=similarity)
            for record, similarity in raw_results
            if record.id != stored_record.id and similarity >= threshold
        ][:top_k]

        if candidates:
            store._emit_event(
                "memory.contradiction_flagged",
                {
                    "record_id": stored_record.id,
                    "memory_type": stored_record.memory_type,
                    "threshold": threshold,
                    "candidate_count": len(candidates),
                    "candidate_ids": [candidate.record.id for candidate in candidates],
                    "top_similarity": candidates[0].similarity,
                },
            )
        return candidates

    def resolve_supersession(
        self,
        superseded_id: str,
        kept_id: str,
        store: SemanticStore,
    ) -> None:
        superseded_record = store.get_by_id(superseded_id)
        if superseded_record is None:
            raise ValueError(f"Semantic record '{superseded_id}' does not exist")

        kept_record = store.get_by_id(kept_id)
        if kept_record is None:
            raise ValueError(f"Semantic record '{kept_id}' does not exist")

        superseded_record.importance = 0.0
        superseded_record.superseded_by = kept_id
        store.replace(superseded_record)

        kept_record.supersedes = superseded_id
        store.replace(kept_record)

    def find_likely_duplicates_batch(
        self,
        store: SemanticStore,
        threshold: float = 0.95,
    ) -> list[tuple[str, str, float]]:
        pairs: list[tuple[str, str, float]] = []
        seen_pairs: set[tuple[str, str]] = set()

        for record in store.get_all_records(include_embeddings=True):
            if record.embedding is None:
                continue

            matches = store.retrieve_by_vector(record.embedding, top_k=2)
            best_other = next((match for match in matches if match[0].id != record.id), None)
            if best_other is None:
                continue

            other, similarity = best_other
            if similarity < threshold:
                continue

            pair_ids = tuple(sorted((record.id, other.id)))
            if pair_ids in seen_pairs:
                continue

            seen_pairs.add(pair_ids)
            pairs.append((pair_ids[0], pair_ids[1], similarity))

        pairs.sort(key=lambda item: (-item[2], item[0], item[1]))
        return pairs
