import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rbj.environment.grid_world import GridWorldEnvironment
from rbj.world import WorldModel


def main():

    print("=" * 60)
    print("ENVIRONMENT + WORLD MODEL")
    print("=" * 60)

    # ==========================================================
    # ENTORNO
    # ==========================================================

    environment = GridWorldEnvironment()

    # ==========================================================
    # CEREBRO DEL AGENTE
    # ==========================================================

    world_model = WorldModel()

    # ==========================================================
    # RESET
    # ==========================================================

    observation = environment.reset()

    print("\nObservación inicial:")
    print(observation)

    # ==========================================================
    # WORLD MODEL RECIBE OBSERVACIÓN
    # ==========================================================

    world_model.update_environment(
        observation["environment"]
    )

    world_model.update_from_perception(
        observation
    )

    print("\nWorld Model:")
    print(world_model.to_dict())

    # ==========================================================
    # ACCIÓN
    # ==========================================================

    action = {
        "action": "MOVE",
        "direction": "RIGHT",
        "distance": 1,
    }

    print("\nAcción:")
    print(action)

    # ==========================================================
    # LA ACCIÓN VA AL ENTORNO
    # ==========================================================

    observation = environment.step(
        action
    )

    print("\nNueva observación:")
    print(observation)

    # ==========================================================
    # WORLD MODEL SE ACTUALIZA
    # ==========================================================

    world_model.update_from_perception(
        observation
    )

    print("\nNuevo World Model:")
    print(world_model.to_dict())

    # ==========================================================
    # POSICIÓN
    # ==========================================================

    print(
        "\nPosición conocida por el agente:",
        world_model.get_robot_position()
    )

    print(
        "\n¿Terminó el entorno?",
        environment.is_done()
    )

    print("\n" + "=" * 60)
    print("TEST COMPLETADO")
    print("=" * 60)


if __name__ == "__main__":
    main()