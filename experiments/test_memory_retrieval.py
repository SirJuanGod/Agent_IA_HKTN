import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rbj.memory import (
    WorkingMemory,
    EpisodicMemory,
    SemanticMemory,
    MemoryRetrieval,
)


def main():

    print("=" * 70)
    print("SJG-AGENT — MEMORY RETRIEVAL TEST")
    print("=" * 70)

    # =================================================
    # MEMORIES
    # =================================================

    working = WorkingMemory(
        capacity=20
    )

    episodic = EpisodicMemory(
        capacity=100
    )

    semantic = SemanticMemory(
        capacity=100
    )

    # =================================================
    # WORKING MEMORY
    # =================================================

    working.set_state(
        "robot_position",
        (2, 3),
    )

    working.set_state(
        "robot_status",
        "IDLE",
    )

    working.set_current_command(
        {
            "action": "MOVE",
            "direction": "LEFT",
            "distance": 2,
        }
    )

    # =================================================
    # SEMANTIC MEMORY
    # =================================================

    semantic.add(
        subject="ROBOT",
        relation="TYPE",
        value="BIPED",
        confidence=1.0,
        source="SYSTEM",
    )

    semantic.add(
        subject="ROBOT",
        relation="DOF",
        value=13,
        confidence=1.0,
        source="SYSTEM",
    )

    semantic.add(
        subject="GOAL",
        relation="POSITION",
        value=(4, 4),
        confidence=1.0,
        source="PERCEPTION",
    )

    semantic.add(
        subject="ENVIRONMENT",
        relation="GRID_SIZE",
        value=(5, 5),
        confidence=1.0,
        source="SYSTEM",
    )

    # =================================================
    # EPISODIC MEMORY
    # =================================================

    episodic.add_episode(

        state_before={
            "position": (2, 3)
        },

        command={
            "action": "MOVE",
            "direction": "RIGHT",
            "distance": 1,
        },

        action={
            "action": "MOVE",
            "direction": "RIGHT",
        },

        state_after={
            "position": (3, 3)
        },

        reward=1.0,

        success=True,

        result="ACTION_COMPLETED",
    )

    episodic.add_episode(

        state_before={
            "position": (3, 3)
        },

        command={
            "action": "MOVE",
            "direction": "UP",
            "distance": 1,
        },

        action={
            "action": "MOVE",
            "direction": "UP",
        },

        state_after={
            "position": (3, 4)
        },

        reward=-1.0,

        success=False,

        result="COLLISION",
    )

    episodic.add_episode(

        state_before={
            "position": (3, 3)
        },

        command={
            "action": "GO_TO",
            "target": "GOAL",
        },

        action={
            "action": "GO_TO",
            "target": "GOAL",
        },

        state_after={
            "position": (4, 4)
        },

        reward=10.0,

        success=True,

        result="GOAL_REACHED",
    )

    # =================================================
    # RETRIEVAL
    # =================================================

    retrieval = MemoryRetrieval(
        working_memory=working,
        episodic_memory=episodic,
        semantic_memory=semantic,
    )

    # =================================================
    # SEMANTIC QUERY
    # =================================================

    print("\n1. SEMANTIC QUERY")
    print("-" * 70)

    results = retrieval.retrieve(
        query={
            "subject": "ROBOT",
            "relation": "DOF",
        }
    )

    for result in results:

        print(result)

    # =================================================
    # EPISODIC QUERY
    # =================================================

    print("\n2. EPISODIC QUERY")
    print("-" * 70)

    results = retrieval.retrieve(
        query={
            "action": "MOVE",
            "success": True,
        }
    )

    for result in results:

        print(result)

    # =================================================
    # FAILED EPISODES
    # =================================================

    print("\n3. FAILED EPISODES")
    print("-" * 70)

    results = retrieval.retrieve(
        query={
            "success": False,
        }
    )

    for result in results:

        print(result)

    # =================================================
    # WORKING MEMORY
    # =================================================

    print("\n4. WORKING MEMORY")
    print("-" * 70)

    results = retrieval.retrieve(
        query={
            "working_key": "robot_position",
        }
    )

    for result in results:

        print(result)

    # =================================================
    # COMBINED QUERY
    # =================================================

    print("\n5. COMBINED QUERY")
    print("-" * 70)

    results = retrieval.retrieve(
        query={
            "subject": "GOAL",
            "relation": "POSITION",
            "action": "GO_TO",
        },
        top_k=10,
    )

    for result in results:

        print(result)

    # =================================================
    # CONTEXT
    # =================================================

    print("\n6. CONTEXT")
    print("-" * 70)

    context = retrieval.build_context(
        query={
            "action": "GO_TO",
        },
        top_k=5,
    )

    print(context)


if __name__ == "__main__":
    main()