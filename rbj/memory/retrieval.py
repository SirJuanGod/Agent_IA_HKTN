from typing import Any, Dict, List, Optional


class MemoryRetrieval:
    """
    Sistema básico de recuperación de memoria.

    Busca información relevante en:

        - Working Memory
        - Episodic Memory
        - Semantic Memory

    La versión inicial utiliza coincidencias simbólicas.
    No utiliza embeddings ni modelos preentrenados.
    """

    def __init__(
        self,
        working_memory=None,
        episodic_memory=None,
        semantic_memory=None,
    ):

        self.working_memory = working_memory
        self.episodic_memory = episodic_memory
        self.semantic_memory = semantic_memory

    # -------------------------------------------------
    # SEMANTIC RETRIEVAL
    # -------------------------------------------------

    def retrieve_semantic(
        self,
        subject: Optional[str] = None,
        relation: Optional[str] = None,
        value: Any = None,
        top_k: int = 5,
    ):

        if self.semantic_memory is None:
            return []

        results = self.semantic_memory.find(
            subject=subject,
            relation=relation,
            value=value,
        )

        results = results[-top_k:]

        return [
            {
                "memory_type": "SEMANTIC",
                "score": 1.0,
                "data": item.to_dict(),
            }
            for item in reversed(results)
        ]

    # -------------------------------------------------
    # EPISODIC RETRIEVAL
    # -------------------------------------------------

    def retrieve_episodes(
        self,
        action: Optional[str] = None,
        success: Optional[bool] = None,
        result: Optional[str] = None,
        top_k: int = 5,
    ):

        if self.episodic_memory is None:
            return []

        episodes = self.episodic_memory.all()

        scored = []

        for episode in episodes:

            score = 0.0

            if action is not None:

                if episode.action:

                    if (
                        episode.action.get("action")
                        == action
                    ):
                        score += 1.0

            if success is not None:

                if episode.success == success:
                    score += 0.5

            if result is not None:

                if episode.result == result:
                    score += 0.5

            if score > 0:

                scored.append(
                    {
                        "memory_type": "EPISODIC",
                        "score": score,
                        "data": episode.to_dict(),
                    }
                )

        scored.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return scored[:top_k]

    # -------------------------------------------------
    # WORKING MEMORY
    # -------------------------------------------------

    def retrieve_working(
        self,
        key: Optional[str] = None,
        top_k: int = 5,
    ):

        if self.working_memory is None:
            return []

        results = []

        if key is not None:

            value = self.working_memory.get(key)

            if value is not None:

                results.append(
                    {
                        "memory_type": "WORKING",
                        "score": 1.0,
                        "data": {
                            "key": key,
                            "value": value,
                        },
                    }
                )

        else:

            recent = self.working_memory.recent(
                top_k
            )

            for item in recent:

                results.append(
                    {
                        "memory_type": "WORKING",
                        "score": 1.0,
                        "data": item,
                    }
                )

        return results[:top_k]

    # -------------------------------------------------
    # CROSS-MEMORY RETRIEVAL
    # -------------------------------------------------

    def retrieve(
        self,
        query: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ):

        query = query or {}

        results = []

        # ---------------------------------------------
        # WORKING
        # ---------------------------------------------

        if (
            "working_key" in query
            or query.get("include_working", False)
        ):

            results.extend(
                self.retrieve_working(
                    key=query.get(
                        "working_key"
                    ),
                    top_k=top_k,
                )
            )

        # ---------------------------------------------
        # SEMANTIC
        # ---------------------------------------------

        if (
            "subject" in query
            or "relation" in query
            or "value" in query
        ):

            results.extend(
                self.retrieve_semantic(
                    subject=query.get(
                        "subject"
                    ),
                    relation=query.get(
                        "relation"
                    ),
                    value=query.get(
                        "value"
                    ),
                    top_k=top_k,
                )
            )

        # ---------------------------------------------
        # EPISODIC
        # ---------------------------------------------

        if (
            "action" in query
            or "success" in query
            or "result" in query
        ):

            results.extend(
                self.retrieve_episodes(
                    action=query.get(
                        "action"
                    ),
                    success=query.get(
                        "success"
                    ),
                    result=query.get(
                        "result"
                    ),
                    top_k=top_k,
                )
            )

        # ---------------------------------------------
        # SORT
        # ---------------------------------------------

        results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return results[:top_k]

    # -------------------------------------------------
    # RELEVANCE SCORE
    # -------------------------------------------------

    def score_match(
        self,
        query: Dict[str, Any],
        memory: Dict[str, Any],
    ):

        score = 0.0

        for key, value in query.items():

            if key in memory:

                if memory[key] == value:
                    score += 1.0

        return score

    # -------------------------------------------------
    # CONTEXT RETRIEVAL
    # -------------------------------------------------

    def build_context(
        self,
        query: Dict[str, Any],
        top_k: int = 5,
    ):

        memories = self.retrieve(
            query=query,
            top_k=top_k,
        )

        context = {
            "query": query,
            "count": len(memories),
            "memories": memories,
        }

        return context