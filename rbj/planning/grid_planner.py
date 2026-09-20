from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from .path_planner import PathPlanner


class GridPathPlanner(PathPlanner):

    DIRECTIONS = {
        "UP": (0, -1),
        "DOWN": (0, 1),
        "LEFT": (-1, 0),
        "RIGHT": (1, 0),
    }

    def find_path(
        self,
        start: List[int],
        goal: List[int],
        world: Dict[str, Any],
    ) -> List[List[int]]:

        environment = world.get("environment", {})

        width = environment.get("width")
        height = environment.get("height")

        if width is None or height is None:
            return []

        obstacles = self._get_obstacles(world)

        start_tuple = tuple(start)
        goal_tuple = tuple(goal)

        if start_tuple == goal_tuple:
            return [list(start_tuple)]

        if start_tuple in obstacles:
            return []

        if goal_tuple in obstacles:
            return []

        queue = deque([start_tuple])

        parent: Dict[
            Tuple[int, int],
            Optional[Tuple[int, int]]
        ] = {
            start_tuple: None
        }

        while queue:

            current = queue.popleft()

            for dx, dy in self.DIRECTIONS.values():

                next_position = (
                    current[0] + dx,
                    current[1] + dy,
                )

                if not self._is_valid_position(
                    next_position,
                    width,
                    height,
                ):
                    continue

                if next_position in obstacles:
                    continue

                if next_position in parent:
                    continue

                parent[next_position] = current

                if next_position == goal_tuple:
                    return self._reconstruct_path(
                        parent,
                        goal_tuple,
                    )

                queue.append(next_position)

        return []

    def _get_obstacles(
        self,
        world: Dict[str, Any],
    ) -> set:

        entities = world.get("entities", {})

        obstacles = set()

        for entity in entities.values():

            if entity.get("entity_type") != "OBSTACLE":
                continue

            position = entity.get(
                "properties",
                {},
            ).get("position")

            if position is not None:
                obstacles.add(tuple(position))

        return obstacles

    @staticmethod
    def _is_valid_position(
        position: Tuple[int, int],
        width: int,
        height: int,
    ) -> bool:

        x, y = position

        return (
            0 <= x < width
            and
            0 <= y < height
        )

    @staticmethod
    def _reconstruct_path(
        parent: Dict[
            Tuple[int, int],
            Optional[Tuple[int, int]]
        ],
        goal: Tuple[int, int],
    ) -> List[List[int]]:

        path = []

        current = goal

        while current is not None:

            path.append(list(current))

            current = parent[current]

        path.reverse()

        return path

    @classmethod
    def path_to_directions(
        cls,
        path: List[List[int]],
    ) -> List[str]:

        directions = []

        for current, next_position in zip(
            path,
            path[1:],
        ):

            dx = next_position[0] - current[0]
            dy = next_position[1] - current[1]

            direction = None

            for name, (
                direction_x,
                direction_y,
            ) in cls.DIRECTIONS.items():

                if (
                    dx == direction_x
                    and
                    dy == direction_y
                ):
                    direction = name
                    break

            if direction is None:
                raise ValueError(
                    f"Movimiento inválido: "
                    f"{current} -> {next_position}"
                )

            directions.append(direction)

        return directions