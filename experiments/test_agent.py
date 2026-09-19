import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

from rbj.environment.grid_world import GridWorld
from rbj.learning.dqn import DQN


def main():

    env = GridWorld(
        size=5,
        max_steps=50
    )

    brain = DQN(
        state_size=4,
        action_size=4
    )

    brain.load(
        "checkpoints/sjg_agent_01.pt"
    )

    state = env.reset()

    done = False

    total_reward = 0

    print("\nEstado inicial:")
    env.render()

    while not done:

        action = brain.choose_action(
            state,
            epsilon=0.0
        )

        print(
            "Action:",
            action
        )

        state, reward, done = env.step(
            action
        )

        total_reward += reward

        env.render()

    print(
        "Reward total:",
        total_reward
    )


if __name__ == "__main__":
    main()