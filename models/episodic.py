from dataclasses import dataclass, field

from models.base import MemoryRecord


@dataclass(kw_only=True)
class EpisodicMemory(MemoryRecord):
    memory_type: str = "episodic"
    session_id: str = "default"
    turn_number: int | None = None
    participants: list[str] = field(default_factory=list)
    summary: str | None = None
    emotional_valence: float | None = None
    source_mime_type: str | None = None

