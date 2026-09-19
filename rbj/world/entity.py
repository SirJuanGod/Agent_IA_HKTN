from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class Entity:
    """
    Representa una entidad dentro del mundo.

    Ejemplos:
        ROBOT
        GOAL
        OBSTACLE_1
        OBSTACLE_2
    """

    entity_id: str
    entity_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    relations: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    def set_property(self, key: str, value: Any) -> None:
        """Establece o actualiza una propiedad."""
        self.properties[key] = value
        self.timestamp = datetime.now().isoformat()

    def get_property(
        self,
        key: str,
        default: Optional[Any] = None
    ) -> Any:
        """Obtiene una propiedad."""
        return self.properties.get(key, default)

    def remove_property(self, key: str) -> None:
        """Elimina una propiedad."""
        self.properties.pop(key, None)
        self.timestamp = datetime.now().isoformat()

    def set_relation(self, relation: str, value: Any) -> None:
        """Establece una relación con otra entidad."""
        self.relations[relation] = value
        self.timestamp = datetime.now().isoformat()

    def get_relation(
        self,
        relation: str,
        default: Optional[Any] = None
    ) -> Any:
        """Obtiene una relación."""
        return self.relations.get(relation, default)

    def remove_relation(self, relation: str) -> None:
        """Elimina una relación."""
        self.relations.pop(relation, None)
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convierte la entidad a un diccionario."""
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "properties": self.properties.copy(),
            "relations": self.relations.copy(),
            "metadata": self.metadata.copy(),
            "timestamp": self.timestamp,
        }