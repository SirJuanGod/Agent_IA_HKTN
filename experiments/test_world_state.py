import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rbj.world import Entity, WorldState


def main():

    print("=" * 60)
    print("TEST WORLD STATE")
    print("=" * 60)

    world = WorldState()

    # ----------------------------------------------------------
    # ENTORNO
    # ----------------------------------------------------------

    world.set_environment("width", 5)
    world.set_environment("height", 5)
    world.set_environment("type", "GRID_WORLD")

    # ----------------------------------------------------------
    # ROBOT
    # ----------------------------------------------------------

    robot = Entity(
        entity_id="ROBOT",
        entity_type="AGENT",
        properties={
            "position": [2, 3],
            "velocity": [0, 0],
            "status": "ACTIVE",
        },
    )

    world.add_entity(robot)

    # ----------------------------------------------------------
    # GOAL
    # ----------------------------------------------------------

    goal = Entity(
        entity_id="GOAL",
        entity_type="GOAL",
        properties={
            "position": [4, 4],
        },
    )

    world.add_entity(goal)

    # ----------------------------------------------------------
    # OBSTACLES
    # ----------------------------------------------------------

    obstacle_positions = [
        [1, 1],
        [2, 1],
        [3, 3],
    ]

    for index, position in enumerate(obstacle_positions):

        obstacle = Entity(
            entity_id=f"OBSTACLE_{index + 1}",
            entity_type="OBSTACLE",
            properties={
                "position": position,
            },
        )

        world.add_entity(obstacle)

    # ----------------------------------------------------------
    # RELACIONES
    # ----------------------------------------------------------

    world.set_relation(
        "ROBOT",
        "HAS_GOAL",
        "GOAL",
    )

    world.set_relation(
        "ROBOT",
        "LOCATED_IN",
        "GRID_WORLD",
    )

    # ----------------------------------------------------------
    # ACTUALIZACIÓN
    # ----------------------------------------------------------

    world.set_entity_property(
        "ROBOT",
        "position",
        [2, 4],
    )

    # ----------------------------------------------------------
    # MOSTRAR
    # ----------------------------------------------------------

    print("\nEstado del entorno:")
    print(world.environment)

    print("\nEntidades:")

    for entity in world.entities.values():

        print(f"\nID: {entity.entity_id}")
        print(f"Tipo: {entity.entity_type}")
        print(f"Propiedades: {entity.properties}")
        print(f"Relaciones: {entity.relations}")

    print("\nResumen:")
    print(world.summary())

    print("\nEstado completo:")
    print(world.to_dict())

    # ----------------------------------------------------------
    # CONSULTAS
    # ----------------------------------------------------------

    robot_position = world.get_entity_property(
        "ROBOT",
        "position",
    )

    print("\nPosición del robot:")
    print(robot_position)

    obstacles = world.get_entities_by_type(
        "OBSTACLE"
    )

    print("\nNúmero de obstáculos:")
    print(len(obstacles))

    goal = world.get_entity("GOAL")

    print("\nPosición de la meta:")
    print(goal.get_property("position"))

    print("\nRelación ROBOT -> HAS_GOAL:")
    print(
        world.get_relation(
            "ROBOT",
            "HAS_GOAL",
        )
    )

    print("\n" + "=" * 60)
    print("TEST COMPLETADO")
    print("=" * 60)


if __name__ == "__main__":
    main()