import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rbj.world import WorldModel


def print_separator(title):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def main():

    print_separator("TEST WORLD MODEL")

    world = WorldModel()

    # ============================================================
    # 1. CREAR ENTORNO
    # ============================================================

    print_separator("1. ENVIRONMENT")

    world.update_environment({
        "type": "GRID_WORLD",
        "width": 5,
        "height": 5,
    })

    print(world.state.environment)

    # ============================================================
    # 2. RECIBIR PERCEPCIÓN
    # ============================================================

    print_separator("2. PERCEPTION")

    perception = {
        "agent": {
            "position": [2, 3],
            "velocity": [0, 0],
            "status": "ACTIVE",
        },

        "goal": {
            "position": [4, 4],
        },

        "obstacles": [
            [1, 1],
            [2, 1],
            [3, 3],
        ],
    }

    world.update_from_perception(
        perception
    )

    print(world.to_dict())

    # ============================================================
    # 3. INFERIR RELACIONES
    # ============================================================

    print_separator("3. RELATIONS")

    world.infer_basic_relations()

    robot = world.state.get_entity("ROBOT")

    print("Robot:")
    print(robot.to_dict())

    # ============================================================
    # 4. CONSULTAR POSICIÓN
    # ============================================================

    print_separator("4. ROBOT POSITION")

    print(
        "Posición:",
        world.get_robot_position()
    )

    print(
        "Meta:",
        world.get_goal_position()
    )

    # ============================================================
    # 5. EJECUTAR MOVIMIENTO
    # ============================================================

    print_separator("5. ACTION")

    action = {
        "action": "MOVE",
        "direction": "RIGHT",
        "distance": 1,
    }

    print("Antes:")
    print(world.get_robot_position())

    world.update_from_action(
        action
    )

    print("Después:")
    print(world.get_robot_position())

    # ============================================================
    # 6. NUEVA PERCEPCIÓN
    # ============================================================

    print_separator("6. NEW PERCEPTION")

    new_perception = {
        "agent": {
            "position": [3, 3],
            "velocity": [1, 0],
            "status": "ACTIVE",
        },

        "goal": {
            "position": [4, 4],
        },

        "obstacles": [
            [1, 1],
            [2, 1],
            [3, 3],
        ],
    }

    world.update_from_perception(
        new_perception
    )

    print(
        "Posición actual:",
        world.get_robot_position()
    )

    # ============================================================
    # 7. OBSTÁCULOS
    # ============================================================

    print_separator("7. OBSTACLES")

    obstacles = world.get_obstacles()

    for obstacle in obstacles:

        print(
            obstacle.entity_id,
            "→",
            obstacle.get_property("position")
        )

    # ============================================================
    # 8. RESUMEN
    # ============================================================

    print_separator("8. SUMMARY")

    print(world.summary())

    print_separator("TEST COMPLETADO")


if __name__ == "__main__":
    main()