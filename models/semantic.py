from dataclasses import dataclass, field

from models.base import MemoryRecord


@dataclass(kw_only=True)
class SemanticMemory(MemoryRecord):
    memory_type: str = "semantic"
    category: str = "general"
    domain: str | None = None
    confidence: float = 1.0
    supersedes: str | None = None               # id of older record this replaces
    related_ids: list[str] = field(default_factory=list)
    has_visual: bool = False
