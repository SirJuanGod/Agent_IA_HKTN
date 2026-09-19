from collections import deque
from typing import Any, Optional

from rbj.memory.memory_item import MemoryItem


class WorkingMemory:
    """
    Memoria de trabajo de SJG-Agent.

    Almacena información temporal y de corto plazo.

    Características:
        - capacidad limitada
        - información reciente
        - estado actual
        - objetivo actual
        - comando actual
        - historial reciente
    """

    def __init__(self, capacity=20):

        if capacity <= 0:
            raise ValueError(
                "capacity debe ser mayor que cero."
            )

        self.capacity = capacity

        self._items = deque(
            maxlen=capacity
        )

        self._state = {}

    # -------------------------------------------------
    # GENERIC MEMORY
    # -------------------------------------------------

    def store(
        self,
        key: str,
        value: Any,
        memory_type: str = "WORKING",
        metadata: Optional[dict] = None,
    ):
        """
        Guarda un elemento en memoria.
        """

        item = MemoryItem(
            memory_type=memory_type,
            key=key,
            value=value,
            metadata=metadata or {},
        )

        self._items.append(item)

        return item

    def get(self, key: str):
        """
        Recupera el elemento más reciente asociado a una clave.
        """

        for item in reversed(self._items):

            if item.key == key:
                return item.value

        return None

    def contains(self, key: str):

        return self.get(key) is not None

    def remove(self, key: str):

        remaining = deque(
            maxlen=self.capacity
        )

        removed = False

        for item in self._items:

            if item.key == key:
                removed = True
            else:
                remaining.append(item)

        self._items = remaining

        return removed

    def clear(self):

        self._items.clear()
        self._state.clear()

    # -------------------------------------------------
    # STATE
    # -------------------------------------------------

    def set_state(self, key: str, value: Any):

        self._state[key] = value

    def get_state(
        self,
        key: str,
        default=None,
    ):

        return self._state.get(
            key,
            default,
        )

    def remove_state(self, key: str):

        if key in self._state:
            del self._state[key]

    def get_all_state(self):

        return self._state.copy()

    # -------------------------------------------------
    # COMMAND
    # -------------------------------------------------

    def set_current_command(self, command):

        self.set_state(
            "current_command",
            command,
        )

        self.store(
            key="current_command",
            value=command,
            memory_type="COMMAND",
        )

    def get_current_command(self):

        return self.get_state(
            "current_command"
        )

    def clear_current_command(self):

        self.remove_state(
            "current_command"
        )

    # -------------------------------------------------
    # GOAL
    # -------------------------------------------------

    def set_goal(self, goal):

        self.set_state(
            "active_goal",
            goal,
        )

    def get_goal(self):

        return self.get_state(
            "active_goal"
        )

    def clear_goal(self):

        self.remove_state(
            "active_goal"
        )

    # -------------------------------------------------
    # HISTORY
    # -------------------------------------------------

    def add_history(self, event):

        self.store(
            key="history",
            value=event,
            memory_type="HISTORY",
        )

    def get_history(self):

        return [
            item.value
            for item in self._items
            if item.key == "history"
        ]

    # -------------------------------------------------
    # RECENT ITEMS
    # -------------------------------------------------

    def recent(self, n=5):

        items = list(self._items)

        return [
            item.to_dict()
            for item in items[-n:]
        ]

    # -------------------------------------------------
    # DEBUG
    # -------------------------------------------------

    def summary(self):

        return {
            "capacity": self.capacity,
            "items": len(self._items),
            "state": self.get_all_state(),
            "history_size": len(
                self.get_history()
            ),
        }