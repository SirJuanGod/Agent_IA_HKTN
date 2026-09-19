from dataclasses import dataclass, field
from typing import Any, Dict
from datetime import datetime


@dataclass
class MemoryItem:
    """
    Elemento almacenado en memoria.

    Representa información temporal que el agente
    puede necesitar durante su funcionamiento.
    """

    memory_type: str
    key: str
    value: Any

    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self):
        return {
            "memory_type": self.memory_type,
            "key": self.key,
            "value": self.value,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }