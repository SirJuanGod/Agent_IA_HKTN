import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

import matplotlib.pyplot as plt

from rbj.environment.grid_world import GridWorld
from rbj.learning.dqn import DQN
from rbj.core.agent import SJGAgent


EPISODES = 1000

TARGET_UPDATE = 20

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995


def main():

    env = GridWorld(
        size=5,
        max_steps=50
    )

    brain = DQN(
        state_size=4,
        action_size=4,
        learning_rate=0.001,
        gamma=0.99
    )

    agent = SJGAgent(
        brain
    )

    epsilon = EPSILON_START

    rewards_history = []

    losses_history = []

    for episode in range(1, EPISODES + 1):

        state = env.reset()

        total_reward = 0

        episode_losses = []

        done = False

        while not done:

            action = agent.act(
                state,
                epsilon
            )

            next_state, reward, done = env.step(
                action
            )

            brain.memory.add(
                state,
                action,
                reward,
                next_state,
                done
            )

            loss = agent.learn()

            if loss is not None:
                episode_losses.append(loss)

            state = next_state

            total_reward += reward

        epsilon = max(
            EPSILON_END,
            epsilon * EPSILON_DECAY
        )

        if episode % TARGET_UPDATE == 0:

            brain.update_target()

        rewards_history.append(
            total_reward
        )

        if episode_losses:

            losses_history.append(
                sum(episode_losses)
                / len(episode_losses)
            )

        else:

            losses_history.append(0)

        if episode % 10 == 0:

            average_reward = sum(
                rewards_history[-10:]
            ) / 10

            print(
                f"Episode: {episode:4d} | "
                f"Reward: {total_reward:7.2f} | "
                f"Average: {average_reward:7.2f} | "
                f"Epsilon: {epsilon:.3f}"
            )

    os.makedirs(
        "checkpoints",
        exist_ok=True
    )

    brain.save(
        "checkpoints/sjg_agent_01.pt"
    )

    print()
    print("Modelo guardado.")
    print()

    plot_results(
        rewards_history,
        losses_history
    )


def plot_results(
    rewards,
    losses
):

    plt.figure()

    plt.plot(
        rewards
    )

    plt.title(
        "SJG-Agent 0.1 - Rewards"
    )

    plt.xlabel(
        "Episode"
    )

    plt.ylabel(
        "Reward"
    )

    plt.grid()

    plt.show()

    plt.figure()

    plt.plot(
        losses
    )

    plt.title(
        "SJG-Agent 0.1 - Loss"
    )

    plt.xlabel(
        "Episode"
    )

    plt.ylabel(
        "Loss"
    )

    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()