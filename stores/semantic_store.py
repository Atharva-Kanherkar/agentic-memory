import chromadb
from datetime import datetime, timezone

from config import CHROMA_DB_PATH
from models.semantic import SemanticMemory
from stores.base import BaseStore
from utils.embeddings import GeminiEmbedder


class SemanticStore(BaseStore):
    """ChromaDB-backed store for semantic (factual) memories."""

    def __init__(self):
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self._collection = client.get_or_create_collection(name="semantic_memories")
        self._embedder = GeminiEmbedder()

    # ── write ──────────────────────────────────────────────────────────────

    def store(self, record: SemanticMemory) -> str:
        embedding = self._embedder.embed_text(record.content)
        record.embedding = embedding

        self._collection.add(
            ids=[record.id],
            embeddings=[embedding],
            documents=[record.content],
            metadatas=[self._to_metadata(record)],
        )
        return record.id

    # ── read ───────────────────────────────────────────────────────────────

    def get_by_id(self, record_id: str) -> SemanticMemory | None:
        result = self._collection.get(
            ids=[record_id],
            include=["embeddings", "documents", "metadatas"],
        )
        if not result["ids"]:
            return None
        return self._from_result(result, 0)

    # ── serialisation helpers ──────────────────────────────────────────────
    # ChromaDB metadata must be flat (str/int/float/bool only — no nested dicts).

    def _to_metadata(self, record: SemanticMemory) -> dict:
        return {
            "memory_type": record.memory_type,
            "modality": record.modality,
            "created_at": record.created_at.isoformat(),
            "importance": record.importance,
            "source": record.source or "",
            "category": record.category,
            "confidence": record.confidence,
        }

    def _from_result(self, result: dict, index: int) -> SemanticMemory:
        meta = result["metadatas"][index]
        return SemanticMemory(
            content=result["documents"][index],
            id=result["ids"][index],
            embedding=result["embeddings"][index],
            created_at=datetime.fromisoformat(meta["created_at"]),
            importance=float(meta["importance"]),
            source=meta["source"] or None,
            category=meta["category"],
            confidence=float(meta["confidence"]),
            modality=meta["modality"],
        )
