from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Observation:
    """
    Observación producida por un entorno.

    Representa información disponible para SJG-Agent,
    no el estado interno completo del simulador.
    """

    environment: Dict[str, Any] = field(
        default_factory=dict
    )

    agent: Dict[str, Any] = field(
        default_factory=dict
    )

    entities: Dict[str, Any] = field(
        default_factory=dict
    )

    sensors: Dict[str, Any] = field(
        default_factory=dict
    )

    raw: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "environment": self.environment,
            "agent": self.agent,
            "entities": self.entities,
            "sensors": self.sensors,
            "raw": self.raw,
        }