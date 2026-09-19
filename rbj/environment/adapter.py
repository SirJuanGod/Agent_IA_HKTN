import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Dict, Optional

from rbj.environment.base import Environment


class EnvironmentAdapter:
    """
    Interfaz única entre SJG-Agent y un entorno externo.

    El agente no necesita conocer la implementación concreta
    del entorno.

    Puede trabajar con:

        GridWorld
        MuJoCo
        Genesis
        PyBullet
        ROS
        robot físico
        etc.
    """

    def __init__(
        self,
        environment: Environment
    ):
        if not isinstance(
            environment,
            Environment
        ):
            raise TypeError(
                "environment debe implementar "
                "la interfaz Environment."
            )

        self.environment = environment

    # ============================================================
    # RESET
    # ============================================================

    def reset(self) -> Dict[str, Any]:
        """
        Reinicia el entorno.
        """

        observation = self.environment.reset()

        return self._validate_observation(
            observation
        )

    # ============================================================
    # OBSERVE
    # ============================================================

    def observe(self) -> Dict[str, Any]:
        """
        Obtiene la observación actual.
        """

        observation = self.environment.observe()

        return self._validate_observation(
            observation
        )

    # ============================================================
    # STEP
    # ============================================================

    def step(
        self,
        action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Envía una acción al entorno.
        """

        if not isinstance(
            action,
            dict
        ):
            raise TypeError(
                "action debe ser un diccionario."
            )

        observation = self.environment.step(
            action
        )

        return self._validate_observation(
            observation
        )

    # ============================================================
    # DONE
    # ============================================================

    def is_done(self) -> bool:
        """
        Indica si el entorno terminó.
        """

        return bool(
            self.environment.is_done()
        )

    # ============================================================
    # CLOSE
    # ============================================================

    def close(self) -> None:
        """
        Cierra el entorno.
        """

        self.environment.close()

    # ============================================================
    # VALIDATION
    # ============================================================

    @staticmethod
    def _validate_observation(
        observation: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:

        if observation is None:
            return {}

        if not isinstance(
            observation,
            dict
        ):
            raise TypeError(
                "El entorno debe devolver "
                "una observación tipo dict."
            )

        return observation

    # ============================================================
    # INFORMATION
    # ============================================================

    def get_environment(self) -> Environment:
        """
        Devuelve el entorno subyacente.

        Se utilizará principalmente para debugging,
        configuración o acceso avanzado.
        """

        return self.environment