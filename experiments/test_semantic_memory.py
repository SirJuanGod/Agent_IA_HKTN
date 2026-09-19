import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rbj.memory import SemanticMemory


def main():

    memory = SemanticMemory(
        capacity=100
    )

    print("=" * 70)
    print("SJG-AGENT — SEMANTIC MEMORY TEST")
    print("=" * 70)

    # -------------------------------------------------
    # KNOWLEDGE
    # -------------------------------------------------

    memory.add(
        subject="ROBOT",
        relation="TYPE",
        value="BIPED",
        confidence=1.0,
        source="SYSTEM",
    )

    memory.add(
        subject="ROBOT",
        relation="DOF",
        value=13,
        confidence=1.0,
        source="SYSTEM",
    )

    memory.add(
        subject="ROBOT",
        relation="IMU_COUNT",
        value=1,
        confidence=1.0,
        source="SYSTEM",
    )

    memory.add(
        subject="ENVIRONMENT",
        relation="GRID_SIZE",
        value=(5, 5),
        confidence=1.0,
        source="SYSTEM",
    )

    memory.add(
        subject="GOAL",
        relation="POSITION",
        value=(4, 4),
        confidence=1.0,
        source="PERCEPTION",
    )

    memory.add(
        subject="ACTION_MOVE",
        relation="REQUIRES",
        value="DIRECTION",
        confidence=1.0,
        source="LANGUAGE",
    )

    memory.add(
        subject="ACTION_MOVE",
        relation="REQUIRES",
        value="DISTANCE",
        confidence=1.0,
        source="LANGUAGE",
    )

    # -------------------------------------------------
    # ALL KNOWLEDGE
    # -------------------------------------------------

    print("\nCONOCIMIENTO:")

    for item in memory.all():

        print(
            item.to_dict()
        )

    # -------------------------------------------------
    # QUERY
    # -------------------------------------------------

    print("\nCONSULTAS:")

    print(
        "Robot type:",
        memory.query(
            "ROBOT",
            "TYPE",
        ),
    )

    print(
        "Robot DOF:",
        memory.query(
            "ROBOT",
            "DOF",
        ),
    )

    print(
        "Goal position:",
        memory.query(
            "GOAL",
            "POSITION",
        ),
    )

    print(
        "Grid size:",
        memory.query(
            "ENVIRONMENT",
            "GRID_SIZE",
        ),
    )

    # -------------------------------------------------
    # FIND
    # -------------------------------------------------

    print("\nCONOCIMIENTO SOBRE ROBOT:")

    robot_knowledge = memory.find(
        subject="ROBOT"
    )

    for item in robot_knowledge:

        print(
            item.subject,
            "->",
            item.relation,
            "->",
            item.value,
        )

    # -------------------------------------------------
    # CONFIDENCE
    # -------------------------------------------------

    print("\nCONFIANZA:")

    print(
        memory.get_confidence(
            "GOAL",
            "POSITION",
        )
    )

    # -------------------------------------------------
    # UPDATE
    # -------------------------------------------------

    memory.update_confidence(
        "GOAL",
        "POSITION",
        0.85,
    )

    print(
        "Nueva confianza:",
        memory.get_confidence(
            "GOAL",
            "POSITION",
        ),
    )

    # -------------------------------------------------
    # SUMMARY
    # -------------------------------------------------

    print("\nSUMMARY:")

    print(
        memory.summary()
    )


if __name__ == "__main__":
    main()