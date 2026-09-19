from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class Episode:
    """
    Representa una experiencia completa del agente.

    Un episodio contiene:

        - estado antes de la acción
        - comando recibido
        - acción ejecutada
        - resultado
        - estado después de la acción
        - recompensa
        - éxito o fallo
        - información adicional
    """

    episode_id: int

    state_before: Dict[str, Any]

    command: Optional[Dict[str, Any]]

    action: Optional[Dict[str, Any]]

    state_after: Optional[Dict[str, Any]] = None

    reward: Optional[float] = None

    success: Optional[bool] = None

    result: Optional[str] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    def to_dict(self):

        return {
            "episode_id": self.episode_id,
            "timestamp": self.timestamp,
            "state_before": self.state_before,
            "command": self.command,
            "action": self.action,
            "state_after": self.state_after,
            "reward": self.reward,
            "success": self.success,
            "result": self.result,
            "metadata": self.metadata,
        }


class EpisodicMemory:
    """
    Memoria episódica de SJG-Agent.

    Almacena experiencias completas y permite
    recuperarlas posteriormente.
    """

    def __init__(self, capacity=1000):

        if capacity <= 0:
            raise ValueError(
                "capacity debe ser mayor que cero."
            )

        self.capacity = capacity

        self._episodes: List[Episode] = []

        self._next_id = 0

    # -------------------------------------------------
    # STORE
    # -------------------------------------------------

    def add_episode(
        self,
        state_before,
        command,
        action,
        state_after=None,
        reward=None,
        success=None,
        result=None,
        metadata=None,
    ):

        episode = Episode(
            episode_id=self._next_id,
            state_before=state_before,
            command=command,
            action=action,
            state_after=state_after,
            reward=reward,
            success=success,
            result=result,
            metadata=metadata or {},
        )

        self._next_id += 1

        self._episodes.append(episode)

        # Mantener capacidad máxima
        if len(self._episodes) > self.capacity:
            self._episodes.pop(0)

        return episode

    # -------------------------------------------------
    # RETRIEVAL
    # -------------------------------------------------

    def get(self, episode_id):

        for episode in self._episodes:

            if episode.episode_id == episode_id:
                return episode

        return None

    def recent(self, n=5):

        if n <= 0:
            return []

        return self._episodes[-n:]

    def all(self):

        return list(self._episodes)

    # -------------------------------------------------
    # FILTERS
    # -------------------------------------------------

    def successful(self):

        return [
            episode
            for episode in self._episodes
            if episode.success is True
        ]

    def failed(self):

        return [
            episode
            for episode in self._episodes
            if episode.success is False
        ]

    def by_action(self, action_name):

        results = []

        for episode in self._episodes:

            if not episode.action:
                continue

            if (
                episode.action.get("action")
                == action_name
            ):
                results.append(episode)

        return results

    def by_command(self, command_action):

        results = []

        for episode in self._episodes:

            if not episode.command:
                continue

            if (
                episode.command.get("action")
                == command_action
            ):
                results.append(episode)

        return results

    # -------------------------------------------------
    # STATISTICS
    # -------------------------------------------------

    def count(self):

        return len(self._episodes)

    def success_rate(self):

        completed = [
            episode
            for episode in self._episodes
            if episode.success is not None
        ]

        if not completed:
            return 0.0

        successful = sum(
            episode.success
            for episode in completed
        )

        return successful / len(completed)

    def average_reward(self):

        rewards = [
            episode.reward
            for episode in self._episodes
            if episode.reward is not None
        ]

        if not rewards:
            return 0.0

        return sum(rewards) / len(rewards)

    # -------------------------------------------------
    # SEARCH
    # -------------------------------------------------

    def search(self, key, value):

        results = []

        for episode in self._episodes:

            data = episode.to_dict()

            if data.get(key) == value:
                results.append(episode)

        return results

    # -------------------------------------------------
    # SERIALIZATION
    # -------------------------------------------------

    def to_list(self):

        return [
            episode.to_dict()
            for episode in self._episodes
        ]

    # -------------------------------------------------
    # CLEAR
    # -------------------------------------------------

    def clear(self):

        self._episodes.clear()

        self._next_id = 0

    # -------------------------------------------------
    # SUMMARY
    # -------------------------------------------------

    def summary(self):

        return {
            "capacity": self.capacity,
            "episodes": len(self._episodes),
            "successful": len(
                self.successful()
            ),
            "failed": len(
                self.failed()
            ),
            "success_rate": self.success_rate(),
            "average_reward": self.average_reward(),
        }