from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ReasoningContext:
    world: Dict[str, Any]
    command: Optional[Dict[str, Any]]
    memory: Optional[Dict[str, Any]] = None

    def get_robot_position(self):
        entities = self.world.get("entities", {})

        robot = entities.get("ROBOT")

        if not robot:
            return None

        return robot.get("properties", {}).get("position")

    def get_goal_position(self):
        entities = self.world.get("entities", {})

        goal = entities.get("GOAL")

        if not goal:
            return None

        return goal.get("properties", {}).get("position")

    def get_obstacles(self):
        entities = self.world.get("entities", {})

        obstacles = []

        for entity_id, entity in entities.items():
            if entity.get("entity_type") != "OBSTACLE":
                continue

            position = entity.get("properties", {}).get("position")

            if position is not None:
                obstacles.append(position)

        return obstacles

    def get_environment(self):
        return self.world.get("environment", {})

    def get_command(self, key: str, default=None):
        if not self.command:
            return default

        return self.command.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "world": self.world,
            "command": self.command,
            "memory": self.memory,
        }