import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rbj.environment import (
    GridWorldEnvironment,
    EnvironmentAdapter,
)

from rbj.world import WorldModel
from rbj.memory import MemoryModule
from rbj.reasoning import Reasoner
from rbj.planning import Planner
from rbj.core import CognitiveLoop


def main():

    environment = GridWorldEnvironment(
        width=5,
        height=5,
    )

    adapter = EnvironmentAdapter(
        environment
    )

    world_model = WorldModel()

    memory = MemoryModule()

    reasoner = Reasoner()

    planner = Planner()

    loop = CognitiveLoop(
        environment=adapter,
        world_model=world_model,
        memory=memory,
        reasoner=reasoner,
        planner=planner,
    )

    # --------------------------------------------------
    # RESET
    # --------------------------------------------------

    loop.reset()

    print("\n=== WORLD ===")
    print(loop.get_world())

    # --------------------------------------------------
    # GO TO GOAL
    # --------------------------------------------------

    command = {
        "intent": "GO_TO",
        "action": "GO_TO",
        "direction": "NONE",
        "target": "GOAL",
        "distance": 0,
    }

    plan = loop.plan(command)

    print("\n=== PLAN ===")
    print(plan.to_dict())

    print("\n=== STEPS ===")

    for index, step in enumerate(
        plan.steps,
        start=1,
    ):

        print(
            f"{index}. "
            f"{step.action} "
            f"{step.direction}"
        )

    # --------------------------------------------------
    # MEMORY
    # --------------------------------------------------

    print("\n=== MEMORY ===")
    print(loop.get_memory_summary())

    loop.close()


if __name__ == "__main__":
    main()