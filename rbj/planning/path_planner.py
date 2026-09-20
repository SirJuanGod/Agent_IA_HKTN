from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class PathPlanner(ABC):

    @abstractmethod
    def find_path(
        self,
        start: List[int],
        goal: List[int],
        world: Dict[str, Any],
    ) -> List[List[int]]:
        """
        Returns a sequence of positions from start to goal.

        The first position should be start.
        The last position should be goal.
        """
        raise NotImplementedError