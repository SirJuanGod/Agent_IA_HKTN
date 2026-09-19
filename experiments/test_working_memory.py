import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rbj.memory import WorkingMemory


def main():

    memory = WorkingMemory(
        capacity=5
    )

    print("=" * 60)
    print("SJG-AGENT — WORKING MEMORY TEST")
    print("=" * 60)

    # ---------------------------------------------
    # ESTADO
    # ---------------------------------------------

    memory.set_state(
        "robot_position",
        (2, 3),
    )

    memory.set_state(
        "robot_status",
        "IDLE",
    )

    print("\nESTADO:")
    print(memory.get_all_state())

    # ---------------------------------------------
    # COMANDO
    # ---------------------------------------------

    command = {
        "action": "MOVE",
        "direction": "LEFT",
        "distance": 3,
    }

    memory.set_current_command(
        command
    )

    print("\nCOMANDO ACTUAL:")
    print(
        memory.get_current_command()
    )

    # ---------------------------------------------
    # OBJETIVO
    # ---------------------------------------------

    goal = {
        "type": "POSITION",
        "value": (0, 3),
    }

    memory.set_goal(goal)

    print("\nOBJETIVO:")
    print(memory.get_goal())

    # ---------------------------------------------
    # HISTORIAL
    # ---------------------------------------------

    memory.add_history({
        "event": "COMMAND_RECEIVED",
        "command": command,
    })

    memory.add_history({
        "event": "PLANNER_STARTED",
    })

    memory.add_history({
        "event": "ACTION_EXECUTED",
        "action": "MOVE_LEFT",
    })

    print("\nHISTORIAL:")

    for event in memory.get_history():
        print(event)

    # ---------------------------------------------
    # RECENT
    # ---------------------------------------------

    print("\nMEMORIA RECIENTE:")

    for item in memory.recent():
        print(item)

    # ---------------------------------------------
    # SUMMARY
    # ---------------------------------------------

    print("\nSUMMARY:")
    print(memory.summary())

    # ---------------------------------------------
    # CLEAR
    # ---------------------------------------------

    memory.clear_current_command()
    memory.clear_goal()

    print("\nDESPUÉS DE LIMPIAR:")

    print(
        "Command:",
        memory.get_current_command(),
    )

    print(
        "Goal:",
        memory.get_goal(),
    )


if __name__ == "__main__":
    main()