import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Dict, List, Optional

from rbj.world.entity import Entity
from rbj.world.state import WorldState


class WorldModel:
    """
    Representación interna del mundo utilizada por SJG-Agent.

    NO simula física.
    NO ejecuta acciones.
    NO controla robots.

    Recibe observaciones y construye una representación
    estructurada del mundo.
    """

    def __init__(
        self,
        world_state: Optional[WorldState] = None
    ):
        self.state = world_state or WorldState()

    # ============================================================
    # ENVIRONMENT INFORMATION
    # ============================================================

    def update_environment(
        self,
        environment: Dict[str, Any]
    ) -> None:

        if not isinstance(environment, dict):
            raise TypeError(
                "environment debe ser un diccionario."
            )

        for key, value in environment.items():
            self.state.set_environment(
                key,
                value
            )

    # ============================================================
    # PERCEPTION
    # ============================================================

    def update_from_perception(
        self,
        perception: Dict[str, Any]
    ) -> None:
        """
        Actualiza el World Model utilizando una observación.

        La observación puede provenir de:
            - visión
            - IMU
            - encoders
            - sensores
            - simulador
            - robot real
        """

        if not isinstance(perception, dict):
            raise TypeError(
                "perception debe ser un diccionario."
            )

        # --------------------------------------------------------
        # AGENT
        # --------------------------------------------------------

        agent = perception.get("agent")

        if agent is not None:

            if not self.state.has_entity("ROBOT"):

                self._create_entity(
                    entity_id="ROBOT",
                    entity_type="AGENT",
                    properties=agent,
                )

            else:

                self._update_entity(
                    "ROBOT",
                    agent,
                )

        # --------------------------------------------------------
        # GOAL
        # --------------------------------------------------------

        goal = perception.get("goal")

        if goal is not None:

            if not self.state.has_entity("GOAL"):

                self._create_entity(
                    entity_id="GOAL",
                    entity_type="GOAL",
                    properties=goal,
                )

            else:

                self._update_entity(
                    "GOAL",
                    goal,
                )

        # --------------------------------------------------------
        # OBSTACLES
        # --------------------------------------------------------

        obstacles = perception.get(
            "obstacles"
        )

        if obstacles is not None:

            self._update_obstacles(
                obstacles
            )

        # --------------------------------------------------------
        # RELATIONS
        # --------------------------------------------------------

        self._update_relations()

    # ============================================================
    # ENTITY MANAGEMENT
    # ============================================================

    def _create_entity(
        self,
        entity_id: str,
        entity_type: str,
        properties: Dict[str, Any]
    ) -> None:

        entity = Entity(
            entity_id=entity_id,
            entity_type=entity_type,
            properties=properties.copy(),
        )

        self.state.add_entity(
            entity
        )

    def _update_entity(
        self,
        entity_id: str,
        properties: Dict[str, Any]
    ) -> None:

        entity = self.state.get_entity(
            entity_id
        )

        if entity is None:
            return

        for key, value in properties.items():

            entity.set_property(
                key,
                value
            )

    def _update_obstacles(
        self,
        obstacles: List[Any]
    ) -> None:

        # Eliminar representación anterior.
        existing = self.state.get_entities_by_type(
            "OBSTACLE"
        )

        for obstacle in existing:

            self.state.remove_entity(
                obstacle.entity_id
            )

        # Añadir nueva observación.
        for index, obstacle in enumerate(
            obstacles
        ):

            if isinstance(
                obstacle,
                dict
            ):
                properties = obstacle

            else:
                properties = {
                    "position": obstacle
                }

            self._create_entity(
                entity_id=f"OBSTACLE_{index + 1}",
                entity_type="OBSTACLE",
                properties=properties,
            )

    # ============================================================
    # RELATIONS
    # ============================================================

    def _update_relations(self) -> None:

        robot = self.state.get_entity(
            "ROBOT"
        )

        if robot is None:
            return

        goal = self.state.get_entity(
            "GOAL"
        )

        if goal is not None:

            robot.set_relation(
                "HAS_GOAL",
                "GOAL"
            )

        environment_type = (
            self.state.get_environment(
                "type"
            )
        )

        if environment_type is not None:

            robot.set_relation(
                "LOCATED_IN",
                environment_type
            )

    # ============================================================
    # QUERIES
    # ============================================================

    def get_robot_position(self):

        return self.state.get_entity_property(
            "ROBOT",
            "position"
        )

    def get_goal_position(self):

        return self.state.get_entity_property(
            "GOAL",
            "position"
        )

    def get_obstacles(self):

        return self.state.get_entities_by_type(
            "OBSTACLE"
        )

    # ============================================================
    # STATE
    # ============================================================

    def get_state(self):

        return self.state

    def to_dict(self):

        return self.state.to_dict()

    def summary(self):

        return self.state.summary()

    def reset(self):

        self.state.clear()