import os 
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Dict

from rbj.environment.base import Environment


class GridWorldEnvironment(Environment):
    """
    Entorno mínimo de prueba para SJG-Agent.

    Este componente representa el mundo.
    NO forma parte del World Model.
    """

    def __init__(
        self,
        width: int = 5,
        height: int = 5
    ):

        self.width = width
        self.height = height

        self.agent_position = [0, 0]

        self.goal_position = [4, 4]

        self.obstacles = [
            [1, 1],
            [2, 1],
            [3, 3],
        ]

        self.done = False

    # ============================================================
    # RESET
    # ============================================================

    def reset(self) -> Dict[str, Any]:

        self.agent_position = [0, 0]

        self.done = False

        return self.observe()

    # ============================================================
    # OBSERVE
    # ============================================================

    def observe(self) -> Dict[str, Any]:

        return {
            "environment": {
                "type": "GRID_WORLD",
                "width": self.width,
                "height": self.height,
            },

            "agent": {
                "position": self.agent_position.copy(),
            },

            "goal": {
                "position": self.goal_position.copy(),
            },

            "obstacles": [
                obstacle.copy()
                for obstacle in self.obstacles
            ],
        }

    # ============================================================
    # STEP
    # ============================================================

    def step(
        self,
        action: Dict[str, Any]
    ) -> Dict[str, Any]:

        if self.done:
            return self.observe()

        if action.get("action") != "MOVE":
            return self.observe()

        direction = action.get(
            "direction"
        )

        distance = action.get(
            "distance",
            1
        )

        if not isinstance(
            distance,
            int
        ):
            distance = 1

        x, y = self.agent_position

        if direction == "UP":
            y -= distance

        elif direction == "DOWN":
            y += distance

        elif direction == "LEFT":
            x -= distance

        elif direction == "RIGHT":
            x += distance

        # Mantener dentro del mundo.
        x = max(
            0,
            min(
                self.width - 1,
                x
            )
        )

        y = max(
            0,
            min(
                self.height - 1,
                y
            )
        )

        # No atravesar obstáculos.
        if [x, y] not in self.obstacles:

            self.agent_position = [
                x,
                y
            ]

        if (
            self.agent_position
            == self.goal_position
        ):
            self.done = True

        return self.observe()

    # ============================================================
    # DONE
    # ============================================================

    def is_done(self) -> bool:

        return self.done

    # ============================================================
    # CLOSE
    # ============================================================

    def close(self) -> None:

        pass