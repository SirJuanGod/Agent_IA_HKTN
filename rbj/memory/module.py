import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import Any, Dict, Optional
from rbj.memory.working_memory import WorkingMemory
from rbj.memory.episodic_memory import EpisodicMemory
from rbj.memory.semantic_memory import SemanticMemory
from rbj.memory.retrieval import MemoryRetrieval


class MemoryModule:
    """
    Interfaz unificada de memoria para SJG-Agent.

    Integra:

        - Working Memory
        - Episodic Memory
        - Semantic Memory
        - Memory Retrieval

    El resto del agente no necesita conocer
    los detalles internos de cada memoria.
    """

    def __init__(
        self,
        working_capacity=20,
        episodic_capacity=1000,
        semantic_capacity=5000,
    ):

        self.working = WorkingMemory(
            capacity=working_capacity
        )

        self.episodic = EpisodicMemory(
            capacity=episodic_capacity
        )

        self.semantic = SemanticMemory(
            capacity=semantic_capacity
        )

        self.retrieval = MemoryRetrieval(
            working_memory=self.working,
            episodic_memory=self.episodic,
            semantic_memory=self.semantic,
        )

    # =================================================
    # WORKING MEMORY
    # =================================================

    def set_state(
        self,
        key: str,
        value: Any,
    ):

        self.working.set_state(
            key,
            value,
        )

    def get_state(
        self,
        key: str,
        default=None,
    ):

        return self.working.get_state(
            key,
            default,
        )

    def current_state(self):

        return self.working.get_all_state()

    # =================================================
    # COMMAND
    # =================================================

    def set_command(
        self,
        command: Dict[str, Any],
    ):

        self.working.set_current_command(
            command
        )

    def get_command(self):

        return self.working.get_current_command()

    def clear_command(self):

        self.working.clear_current_command()

    # =================================================
    # GOAL
    # =================================================

    def set_goal(
        self,
        goal: Dict[str, Any],
    ):

        self.working.set_goal(
            goal
        )

    def get_goal(self):

        return self.working.get_goal()

    def clear_goal(self):

        self.working.clear_goal()

    # =================================================
    # EPISODIC MEMORY
    # =================================================

    def remember_episode(
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

        return self.episodic.add_episode(

            state_before=state_before,

            command=command,

            action=action,

            state_after=state_after,

            reward=reward,

            success=success,

            result=result,

            metadata=metadata,
        )

    # =================================================
    # SEMANTIC MEMORY
    # =================================================

    def learn(
        self,
        subject: str,
        relation: str,
        value: Any,
        confidence=1.0,
        source="UNKNOWN",
        metadata: Optional[dict] = None,
    ):

        return self.semantic.add(

            subject=subject,

            relation=relation,

            value=value,

            confidence=confidence,

            source=source,

            metadata=metadata,
        )

    def know(
        self,
        subject: str,
        relation: str,
        default=None,
    ):

        return self.semantic.query(

            subject=subject,

            relation=relation,

            default=default,
        )

    # =================================================
    # RETRIEVAL
    # =================================================

    def recall(
        self,
        query: Dict[str, Any],
        top_k=5,
    ):

        return self.retrieval.retrieve(

            query=query,

            top_k=top_k,
        )

    def context(
        self,
        query: Dict[str, Any],
        top_k=5,
    ):

        return self.retrieval.build_context(

            query=query,

            top_k=top_k,
        )

    # =================================================
    # HISTORY
    # =================================================

    def add_history(
        self,
        event: Dict[str, Any],
    ):

        self.working.add_history(
            event
        )

    def history(self):

        return self.working.get_history()

    # =================================================
    # RECENT MEMORY
    # =================================================

    def recent_working(
        self,
        n=5,
    ):

        return self.working.recent(n)

    def recent_episodes(
        self,
        n=5,
    ):

        return [
            episode.to_dict()
            for episode in self.episodic.recent(n)
        ]

    # =================================================
    # CLEAR
    # =================================================

    def clear_working(self):

        self.working.clear()

    def clear_all(self):

        self.working.clear()

        self.episodic.clear()

        self.semantic.clear()

    # =================================================
    # SUMMARY
    # =================================================

    def summary(self):

        return {
            "working": self.working.summary(),

            "episodic": self.episodic.summary(),

            "semantic": self.semantic.summary(),
        }