from abc import ABC, abstractmethod

from models.base import MemoryRecord


class BaseStore(ABC):
    """All memory stores implement this interface — semantic, episodic, procedural."""

    @abstractmethod
    def store(self, record: MemoryRecord) -> str:
        """Persist a record. Returns its id."""
        ...

    @abstractmethod
    def get_by_id(self, record_id: str) -> MemoryRecord | None:
        """Fetch a single record by id. Returns None if not found."""
        ...
