import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rbj.memory import MemoryModule


def main():

    print("=" * 70)
    print("SJG-AGENT — MEMORY MODULE TEST")
    print("=" * 70)

    memory = MemoryModule(
        working_capacity=20,
        episodic_capacity=100,
        semantic_capacity=100,
    )

    # =================================================
    # 1. CURRENT STATE
    # =================================================

    print("\n1. CURRENT STATE")
    print("-" * 70)

    memory.set_state(
        "robot_position",
        (2, 2),
    )

    memory.set_state(
        "robot_status",
        "IDLE",
    )

    print(
        memory.current_state()
    )

    # =================================================
    # 2. COMMAND
    # =================================================

    print("\n2. COMMAND")
    print("-" * 70)

    command = {
        "action": "MOVE",
        "direction": "LEFT",
        "distance": 2,
    }

    memory.set_command(
        command
    )

    print(
        memory.get_command()
    )

    # =================================================
    # 3. GOAL
    # =================================================

    print("\n3. GOAL")
    print("-" * 70)

    goal = {
        "type": "POSITION",
        "value": (0, 2),
    }

    memory.set_goal(
        goal
    )

    print(
        memory.get_goal()
    )

    # =================================================
    # 4. SEMANTIC KNOWLEDGE
    # =================================================

    print("\n4. SEMANTIC KNOWLEDGE")
    print("-" * 70)

    memory.learn(
        subject="ROBOT",
        relation="TYPE",
        value="BIPED",
        confidence=1.0,
        source="SYSTEM",
    )

    memory.learn(
        subject="ROBOT",
        relation="DOF",
        value=13,
        confidence=1.0,
        source="SYSTEM",
    )

    memory.learn(
        subject="GOAL",
        relation="POSITION",
        value=(4, 4),
        confidence=0.95,
        source="PERCEPTION",
    )

    print(
        "Robot type:",
        memory.know(
            "ROBOT",
            "TYPE",
        ),
    )

    print(
        "Robot DOF:",
        memory.know(
            "ROBOT",
            "DOF",
        ),
    )

    print(
        "Goal:",
        memory.know(
            "GOAL",
            "POSITION",
        ),
    )

    # =================================================
    # 5. EPISODE
    # =================================================

    print("\n5. EPISODE")
    print("-" * 70)

    episode = memory.remember_episode(

        state_before={
            "position": (2, 2),
            "status": "IDLE",
        },

        command=command,

        action={
            "action": "MOVE",
            "direction": "LEFT",
            "distance": 2,
        },

        state_after={
            "position": (0, 2),
            "status": "IDLE",
        },

        reward=1.0,

        success=True,

        result="ACTION_COMPLETED",
    )

    print(
        episode.to_dict()
    )

    # =================================================
    # 6. HISTORY
    # =================================================

    print("\n6. HISTORY")
    print("-" * 70)

    memory.add_history({
        "event": "COMMAND_RECEIVED",
        "command": command,
    })

    memory.add_history({
        "event": "ACTION_EXECUTED",
        "success": True,
    })

    for event in memory.history():

        print(event)

    # =================================================
    # 7. RECALL
    # =================================================

    print("\n7. RECALL")
    print("-" * 70)

    results = memory.recall(
        query={
            "action": "MOVE",
            "success": True,
        },
        top_k=5,
    )

    for result in results:

        print(result)

    # =================================================
    # 8. CONTEXT
    # =================================================

    print("\n8. CONTEXT")
    print("-" * 70)

    context = memory.context(
        query={
            "subject": "GOAL",
            "relation": "POSITION",
        },
        top_k=5,
    )

    print(context)

    # =================================================
    # 9. SUMMARY
    # =================================================

    print("\n9. SUMMARY")
    print("-" * 70)

    print(
        memory.summary()
    )


if __name__ == "__main__":
    main()