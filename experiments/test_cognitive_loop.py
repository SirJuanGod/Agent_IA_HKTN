import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rbj.core import CognitiveLoop

from rbj.environment import (
    EnvironmentAdapter,
    GridWorldEnvironment,
)

from rbj.world import WorldModel

from rbj.memory import MemoryModule


def print_separator(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main():

    print_separator(
        "SJG-AGENT COGNITIVE LOOP"
    )

    # ==========================================================
    # ENVIRONMENT
    # ==========================================================

    environment = EnvironmentAdapter(
        GridWorldEnvironment(
            width=5,
            height=5
        )
    )

    # ==========================================================
    # WORLD MODEL
    # ==========================================================

    world_model = WorldModel()

    # ==========================================================
    # MEMORY
    # ==========================================================

    memory = MemoryModule()

    # ==========================================================
    # COGNITIVE LOOP
    # ==========================================================

    agent = CognitiveLoop(
        environment=environment,
        world_model=world_model,
        memory=memory,
    )

    # ==========================================================
    # RESET
    # ==========================================================

    print_separator(
        "1. RESET"
    )

    observation = agent.reset()

    print(
        "Observation:"
    )

    print(observation)

    print(
        "\nWorld:"
    )

    print(agent.get_world())

    # ==========================================================
    # ACTION 1
    # ==========================================================

    print_separator(
        "2. ACTION 1"
    )

    action = {
        "action": "MOVE",
        "direction": "RIGHT",
        "distance": 1,
    }

    observation = agent.step(
        action
    )

    print(
        "Action:",
        action
    )

    print(
        "Observation:",
        observation
    )

    print(
        "Robot position:",
        world_model.get_robot_position()
    )

    # ==========================================================
    # ACTION 2
    # ==========================================================

    print_separator(
        "3. ACTION 2"
    )

    action = {
        "action": "MOVE",
        "direction": "DOWN",
        "distance": 1,
    }

    observation = agent.step(
        action
    )

    print(
        "Action:",
        action
    )

    print(
        "Observation:",
        observation
    )

    print(
        "Robot position:",
        world_model.get_robot_position()
    )

    # ==========================================================
    # MEMORY
    # ==========================================================

    print_separator(
        "4. MEMORY"
    )

    print(
        "Memory summary:"
    )

    print(
        agent.get_memory_summary()
    )

    # ==========================================================
    # WORLD
    # ==========================================================

    print_separator(
        "5. WORLD MODEL"
    )

    print(
        agent.get_world()
    )

    # ==========================================================
    # STATUS
    # ==========================================================

    print_separator(
        "6. STATUS"
    )

    print(
        "Step:",
        agent.step_count
    )

    print(
        "Done:",
        agent.is_done()
    )

    # ==========================================================
    # CLOSE
    # ==========================================================

    agent.close()

    print_separator(
        "TEST COMPLETED"
    )


if __name__ == "__main__":
    main()