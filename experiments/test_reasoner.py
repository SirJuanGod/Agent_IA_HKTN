import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rbj.environment import (
    GridWorldEnvironment,
    EnvironmentAdapter,
)

from rbj.world import WorldModel
from rbj.memory import MemoryModule
from rbj.reasoning import Reasoner
from rbj.core import CognitiveLoop


def main():

    # --------------------------------------------------
    # ENVIRONMENT
    # --------------------------------------------------

    environment = GridWorldEnvironment(
        width=5,
        height=5,
    )

    adapter = EnvironmentAdapter(
        environment
    )

    # --------------------------------------------------
    # INTERNAL SYSTEMS
    # --------------------------------------------------

    world_model = WorldModel()

    memory = MemoryModule()

    reasoner = Reasoner()

    # --------------------------------------------------
    # COGNITIVE LOOP
    # --------------------------------------------------

    loop = CognitiveLoop(
        environment=adapter,
        world_model=world_model,
        memory=memory,
        reasoner=reasoner,
    )

    # --------------------------------------------------
    # RESET
    # --------------------------------------------------

    observation = loop.reset()

    print("\n=== INITIAL OBSERVATION ===")
    print(observation)

    print("\n=== WORLD ===")
    print(loop.get_world())

    # --------------------------------------------------
    # TEST 1 — MOVE RIGHT
    # --------------------------------------------------

    command = {
        "intent": "MOVE",
        "action": "MOVE",
        "direction": "RIGHT",
        "distance": 1,
        "target": "NONE",
    }

    decision = loop.reason(command)

    print("\n=== DECISION 1 ===")
    print(decision.to_dict())

    # --------------------------------------------------
    # TEST 2 — MOVE UP
    # --------------------------------------------------

    command = {
        "intent": "MOVE",
        "action": "MOVE",
        "direction": "UP",
        "distance": 1,
        "target": "NONE",
    }

    decision = loop.reason(command)

    print("\n=== DECISION 2 ===")
    print(decision.to_dict())

    # --------------------------------------------------
    # TEST 3 — GO TO GOAL
    # --------------------------------------------------

    command = {
        "intent": "GO_TO",
        "action": "GO_TO",
        "direction": "NONE",
        "distance": 0,
        "target": "GOAL",
    }

    decision = loop.reason(command)

    print("\n=== DECISION 3 ===")
    print(decision.to_dict())

    # --------------------------------------------------
    # TEST 4 — STOP
    # --------------------------------------------------

    command = {
        "intent": "STOP",
        "action": "STOP",
        "direction": "NONE",
        "distance": 0,
        "target": "NONE",
    }

    decision = loop.reason(command)

    print("\n=== DECISION 4 ===")
    print(decision.to_dict())

    # --------------------------------------------------
    # MEMORY
    # --------------------------------------------------

    print("\n=== MEMORY ===")
    print(loop.get_memory_summary())

    loop.close()


if __name__ == "__main__":
    main()