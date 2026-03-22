from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import uuid


@dataclass(kw_only=True)
class MemoryRecord:
    content: str                                # required — no default
    memory_type: str                            # required — no default
    modality: str = "text"

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed_at: Optional[datetime] = None
    access_count: int = 0
    importance: float = 0.5

    embedding: Optional[list[float]] = None
    embedding_dims: int = 768

    source: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    media_ref: Optional[str] = None
    media_type: Optional[str] = None
    text_description: Optional[str] = None
