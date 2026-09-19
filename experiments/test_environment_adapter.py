import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rbj.environment import (
    EnvironmentAdapter,
    GridWorldEnvironment,
)


def main():

    print("=" * 60)
    print("TEST ENVIRONMENT ADAPTER")
    print("=" * 60)

    # ----------------------------------------------------------
    # CREAR ENTORNO
    # ----------------------------------------------------------

    grid_world = GridWorldEnvironment(
        width=5,
        height=5
    )

    # ----------------------------------------------------------
    # CREAR ADAPTER
    # ----------------------------------------------------------

    environment = EnvironmentAdapter(
        grid_world
    )

    # ----------------------------------------------------------
    # RESET
    # ----------------------------------------------------------

    observation = environment.reset()

    print("\nObservación inicial:")
    print(observation)

    # ----------------------------------------------------------
    # OBSERVE
    # ----------------------------------------------------------

    observation = environment.observe()

    print("\nObservación actual:")
    print(observation)

    # ----------------------------------------------------------
    # ACTION
    # ----------------------------------------------------------

    action = {
        "action": "MOVE",
        "direction": "RIGHT",
        "distance": 1,
    }

    print("\nAcción:")
    print(action)

    observation = environment.step(
        action
    )

    print("\nNueva observación:")
    print(observation)

    # ----------------------------------------------------------
    # STATUS
    # ----------------------------------------------------------

    print(
        "\n¿Terminó?",
        environment.is_done()
    )

    # ----------------------------------------------------------
    # CLOSE
    # ----------------------------------------------------------

    environment.close()

    print("\n" + "=" * 60)
    print("TEST COMPLETADO")
    print("=" * 60)


if __name__ == "__main__":
    main()