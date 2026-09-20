from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class Knowledge:
    """
    Representa una unidad de conocimiento semántico.

    El conocimiento no depende de una experiencia concreta.
    """

    knowledge_id: int

    subject: str
    relation: str
    value: Any

    confidence: float = 1.0

    source: str = "UNKNOWN"

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    updated_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    def to_dict(self):

        return {
            "knowledge_id": self.knowledge_id,
            "subject": self.subject,
            "relation": self.relation,
            "value": self.value,
            "confidence": self.confidence,
            "source": self.source,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class SemanticMemory:
    """
    Memoria semántica de SJG-Agent.

    Almacena conocimiento reutilizable mediante
    relaciones sujeto -> relación -> valor.

    Ejemplo:

        robot -> type -> MOBILE_ROBOT
        goal -> position -> (4,4)
        environment -> size -> (5,5)
        action_MOVE -> requires -> DIRECTION
    """

    def __init__(self, capacity=5000):

        if capacity <= 0:
            raise ValueError(
                "capacity debe ser mayor que cero."
            )

        self.capacity = capacity

        self._knowledge: deque = deque(maxlen=capacity)

        self._next_id = 0

    # -------------------------------------------------
    # ADD
    # -------------------------------------------------

    def add(
        self,
        subject: str,
        relation: str,
        value: Any,
        confidence: float = 1.0,
        source: str = "UNKNOWN",
        metadata: Optional[dict] = None,
    ):

        if not subject:
            raise ValueError(
                "subject no puede estar vacío."
            )

        if not relation:
            raise ValueError(
                "relation no puede estar vacía."
            )

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "confidence debe estar entre 0 y 1."
            )

        # Si ya existe exactamente el mismo
        # conocimiento, actualizarlo.
        existing = self.find_one(
            subject=subject,
            relation=relation,
        )

        if existing is not None:

            existing.value = value
            existing.confidence = confidence
            existing.source = source
            existing.metadata = metadata or {}
            existing.updated_at = (
                datetime.now().isoformat()
            )

            return existing

        knowledge = Knowledge(
            knowledge_id=self._next_id,
            subject=subject,
            relation=relation,
            value=value,
            confidence=confidence,
            source=source,
            metadata=metadata or {},
        )

        self._next_id += 1

        self._knowledge.append(
            knowledge
        )

        return knowledge

    # -------------------------------------------------
    # GET
    # -------------------------------------------------

    def get(self, knowledge_id):

        for item in self._knowledge:

            if item.knowledge_id == knowledge_id:
                return item

        return None

    # -------------------------------------------------
    # FIND
    # -------------------------------------------------

    def find(
        self,
        subject: Optional[str] = None,
        relation: Optional[str] = None,
        value: Any = None,
    ):

        results = []

        for item in self._knowledge:

            if (
                subject is not None
                and item.subject != subject
            ):
                continue

            if (
                relation is not None
                and item.relation != relation
            ):
                continue

            if (
                value is not None
                and item.value != value
            ):
                continue

            results.append(item)

        return results

    def find_one(
        self,
        subject,
        relation,
    ):

        results = self.find(
            subject=subject,
            relation=relation,
        )

        if not results:
            return None

        return results[-1]

    # -------------------------------------------------
    # QUERY
    # -------------------------------------------------

    def query(
        self,
        subject,
        relation,
        default=None,
    ):

        item = self.find_one(
            subject=subject,
            relation=relation,
        )

        if item is None:
            return default

        return item.value

    # -------------------------------------------------
    # CONFIDENCE
    # -------------------------------------------------

    def get_confidence(
        self,
        subject,
        relation,
    ):

        item = self.find_one(
            subject=subject,
            relation=relation,
        )

        if item is None:
            return 0.0

        return item.confidence

    # -------------------------------------------------
    # UPDATE CONFIDENCE
    # -------------------------------------------------

    def update_confidence(
        self,
        subject,
        relation,
        confidence,
    ):

        item = self.find_one(
            subject=subject,
            relation=relation,
        )

        if item is None:
            return False

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "confidence debe estar entre 0 y 1."
            )

        item.confidence = confidence

        item.updated_at = (
            datetime.now().isoformat()
        )

        return True

    # -------------------------------------------------
    # REMOVE
    # -------------------------------------------------

    def remove(
        self,
        subject,
        relation,
    ):

        original_size = len(
            self._knowledge
        )

        self._knowledge = [
            item
            for item in self._knowledge
            if not (
                item.subject == subject
                and item.relation == relation
            )
        ]

        return (
            len(self._knowledge)
            != original_size
        )

    # -------------------------------------------------
    # ALL
    # -------------------------------------------------

    def all(self):

        return list(self._knowledge)

    def count(self):

        return len(self._knowledge)

    # -------------------------------------------------
    # SERIALIZATION
    # -------------------------------------------------

    def to_list(self):

        return [
            item.to_dict()
            for item in self._knowledge
        ]

    # -------------------------------------------------
    # CLEAR
    # -------------------------------------------------

    def clear(self):

        self._knowledge.clear()

        self._next_id = 0

    # -------------------------------------------------
    # SUMMARY
    # -------------------------------------------------

    def summary(self):

        return {
            "capacity": self.capacity,
            "knowledge_items": len(
                self._knowledge
            ),
        }