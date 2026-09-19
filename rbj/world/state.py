import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import Any, Dict, List, Optional

from rbj.world.entity import Entity


class WorldState:
    """
    Representación simbólica y estructurada del estado del mundo.

    El WorldState no realiza percepción ni razonamiento.
    Solamente mantiene una representación consistente del mundo.
    """

    def __init__(self):
        self.environment: Dict[str, Any] = {}
        self.entities: Dict[str, Entity] = {}

    # ============================================================
    # ENVIRONMENT
    # ============================================================

    def set_environment(self, key: str, value: Any) -> None:
        """Establece una propiedad del entorno."""
        self.environment[key] = value

    def get_environment(
        self,
        key: str,
        default: Optional[Any] = None
    ) -> Any:
        """Obtiene una propiedad del entorno."""
        return self.environment.get(key, default)

    def remove_environment(self, key: str) -> None:
        """Elimina una propiedad del entorno."""
        self.environment.pop(key, None)

    # ============================================================
    # ENTITIES
    # ============================================================

    def add_entity(self, entity: Entity) -> None:
        """Añade una entidad al mundo."""
        if entity.entity_id in self.entities:
            raise ValueError(
                f"La entidad '{entity.entity_id}' ya existe."
            )

        self.entities[entity.entity_id] = entity

    def update_entity(self, entity: Entity) -> None:
        """Actualiza una entidad existente."""
        if entity.entity_id not in self.entities:
            raise KeyError(
                f"La entidad '{entity.entity_id}' no existe."
            )

        self.entities[entity.entity_id] = entity

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Obtiene una entidad por ID."""
        return self.entities.get(entity_id)

    def remove_entity(self, entity_id: str) -> None:
        """Elimina una entidad."""
        self.entities.pop(entity_id, None)

    def has_entity(self, entity_id: str) -> bool:
        """Indica si una entidad existe."""
        return entity_id in self.entities

    def get_entities_by_type(
        self,
        entity_type: str
    ) -> List[Entity]:
        """Obtiene todas las entidades de un determinado tipo."""
        return [
            entity
            for entity in self.entities.values()
            if entity.entity_type == entity_type
        ]

    # ============================================================
    # ENTITY PROPERTIES
    # ============================================================

    def set_entity_property(
        self,
        entity_id: str,
        key: str,
        value: Any
    ) -> None:
        """Actualiza una propiedad de una entidad."""

        entity = self.get_entity(entity_id)

        if entity is None:
            raise KeyError(
                f"La entidad '{entity_id}' no existe."
            )

        entity.set_property(key, value)

    def get_entity_property(
        self,
        entity_id: str,
        key: str,
        default: Optional[Any] = None
    ) -> Any:
        """Obtiene una propiedad de una entidad."""

        entity = self.get_entity(entity_id)

        if entity is None:
            return default

        return entity.get_property(key, default)

    # ============================================================
    # RELATIONS
    # ============================================================

    def set_relation(
        self,
        entity_id: str,
        relation: str,
        value: Any
    ) -> None:
        """Establece una relación para una entidad."""

        entity = self.get_entity(entity_id)

        if entity is None:
            raise KeyError(
                f"La entidad '{entity_id}' no existe."
            )

        entity.set_relation(relation, value)

    def get_relation(
        self,
        entity_id: str,
        relation: str,
        default: Optional[Any] = None
    ) -> Any:
        """Obtiene una relación."""

        entity = self.get_entity(entity_id)

        if entity is None:
            return default

        return entity.get_relation(relation, default)

    # ============================================================
    # SERIALIZATION
    # ============================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convierte todo el estado del mundo a un diccionario."""

        return {
            "environment": self.environment.copy(),
            "entities": {
                entity_id: entity.to_dict()
                for entity_id, entity in self.entities.items()
            },
        }

    # ============================================================
    # SUMMARY
    # ============================================================

    def summary(self) -> Dict[str, Any]:
        """Devuelve un resumen del estado actual."""

        return {
            "environment": self.environment.copy(),
            "num_entities": len(self.entities),
            "entities": [
                {
                    "id": entity.entity_id,
                    "type": entity.entity_type,
                }
                for entity in self.entities.values()
            ],
        }

    # ============================================================
    # CLEAR
    # ============================================================

    def clear(self) -> None:
        """Limpia completamente el estado del mundo."""
        self.environment.clear()
        self.entities.clear()