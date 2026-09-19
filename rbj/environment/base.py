from abc import ABC, abstractmethod
from typing import Any, Dict


class Environment(ABC):
    """
    Interfaz abstracta para cualquier entorno que utilice SJG-Agent.

    Puede representar:
        - GridWorld
        - MuJoCo
        - Genesis
        - PyBullet
        - robot físico
        - simulador propio
    """

    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """
        Reinicia el entorno y devuelve la primera observación.
        """
        raise NotImplementedError

    @abstractmethod
    def observe(self) -> Dict[str, Any]:
        """
        Obtiene la observación actual del entorno.
        """
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Ejecuta una acción en el entorno.

        Devuelve la nueva observación.
        """
        raise NotImplementedError

    @abstractmethod
    def is_done(self) -> bool:
        """
        Indica si el episodio terminó.
        """
        raise NotImplementedError

    def close(self) -> None:
        """
        Libera recursos del entorno.

        No todos los entornos necesitan implementar esto.
        """
        pass