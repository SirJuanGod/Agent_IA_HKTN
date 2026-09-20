"""
Estado interno del agente (stub).

Reservado para encapsular el estado completo del agente en un
objeto serializable, útil para persistencia entre sesiones y
para depuración.

Diseño previsto:
    - AgentState dataclass con campos: step, last_action,
      last_decision, last_plan, world_snapshot, memory_snapshot.
    - Métodos: to_dict(), from_dict(), save(path), load(path).
"""

# TODO: Implementar cuando se requiera persistencia de estado entre sesiones.
