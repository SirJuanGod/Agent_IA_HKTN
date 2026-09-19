import os
import sys
import warnings

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

import torch
import matplotlib.pyplot as plt

from rbj.environment.grid_world import GridWorld
from rbj.learning.dqn import DQN
from rbj.core.agent import SJGAgent
from rbj.models.vision import SJGVision

EPISODES = 1000
TARGET_UPDATE = 20
EPSILON_START = 1.0
EPSILON_END = 0.0005
EPSILON_DECAY = 0.9995

warnings.filterwarnings("ignore", category=DeprecationWarning)

def image_to_tensor(image):
    tensor = torch.tensor(
        list(image.getdata()),
        dtype=torch.float32
    )
    tensor = tensor.reshape(image.height, image.width, 3)
    tensor = tensor.permute(2, 0, 1)
    tensor = tensor / 255.0
    return tensor.unsqueeze(0)

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    env = GridWorld(size=5, max_steps=50)

    vision_model = SJGVision().to(device)
    vision_checkpoint = torch.load("checkpoints/sjg_vision_02.pt", map_location=device)
    if "model_state_dict" in vision_checkpoint:
        vision_model.load_state_dict(vision_checkpoint["model_state_dict"])
    else:
        vision_model.load_state_dict(vision_checkpoint)
    vision_model.eval()

    brain = DQN(state_size=6, action_size=4, learning_rate=0.001, gamma=0.99)
    agent = SJGAgent(brain)

    epsilon = EPSILON_START
    rewards_history = []
    losses_history = []

    for episode in range(1, EPISODES + 1):
        env.reset()
        
        # Generar imagen inicial y extraer estado
        image = env.render_image()
        x = image_to_tensor(image).to(device)
        with torch.no_grad():
            state = vision_model(x)[0]

        total_reward = 0
        episode_losses = []
        done = False

        while not done:
            action = agent.act(state, epsilon)
            _, reward, done = env.step(action)
            
            # Extraer nuevo estado desde la visión
            next_image = env.render_image()
            next_x = image_to_tensor(next_image).to(device)
            with torch.no_grad():
                next_state = vision_model(next_x)[0]

            brain.memory.add(state, action, reward, next_state, done)

            loss = agent.learn()
            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            total_reward += reward

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if episode % TARGET_UPDATE == 0:
            brain.update_target()

        rewards_history.append(total_reward)

        if episode_losses:
            losses_history.append(sum(episode_losses) / len(episode_losses))
        else:
            losses_history.append(0)

        if episode % 10 == 0:
            average_reward = sum(rewards_history[-10:]) / 10
            print(
                f"Episode: {episode:4d} | "
                f"Reward: {total_reward:7.2f} | "
                f"Average: {average_reward:7.2f} | "
                f"Epsilon: {epsilon:.3f}"
            )

    os.makedirs("checkpoints", exist_ok=True)
    brain.save("checkpoints/sjg_vision_agent_01.pt")
    
    print("\nModelo end-to-end guardado.")
    plot_results(rewards_history, losses_history)

def plot_results(rewards, losses):
    plt.figure()
    plt.plot(rewards)
    plt.title("Vision+RL Agent - Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid()
    plt.savefig("vision_agent_rewards.png")
    
    plt.figure()
    plt.plot(losses)
    plt.title("Vision+RL Agent - Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig("vision_agent_loss.png")

if __name__ == "__main__":
    main()
