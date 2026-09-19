import os
import sys
import warnings

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch

from rbj.environment.grid_world import GridWorld
from rbj.learning.dqn import DQN
from rbj.core.agent import SJGAgent
from rbj.models.vision import SJGVision

ACTIONS = {
    0: "↑ UP",
    1: "↓ DOWN",
    2: "← LEFT",
    3: "→ RIGHT"
}


def image_to_tensor(image):
    tensor = torch.tensor(
        list(image.getdata()),
        dtype=torch.float32
    )
    tensor = tensor.reshape(image.height, image.width, 3)
    tensor = tensor.permute(2, 0, 1)
    tensor = tensor / 255.0
    return tensor.unsqueeze(0)


def state_to_readable(state, grid_size=5):
    norm = grid_size - 1
    ax = round(state[0] * norm)
    ay = round(state[1] * norm)
    gx = round(state[2] * norm)
    gy = round(state[3] * norm)
    ox = round(state[4] * norm)
    oy = round(state[5] * norm)
    return (ax, ay), (gx, gy), (ox, oy)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # --- Cargar Visión ---
    vision_model = SJGVision().to(device)
    vision_model.load_state_dict(
        torch.load("checkpoints/sjg_vision_02.pt", map_location=device)
    )
    vision_model.eval()

    # --- Cargar DQN ---
    brain = DQN(state_size=6, action_size=4)
    brain.policy.load_state_dict(
        torch.load("checkpoints/sjg_vision_agent_01.pt", map_location=device)
    )
    brain.policy.eval()

    agent = SJGAgent(brain)

    # --- Entorno ---
    env = GridWorld(size=5, max_steps=50)
    env.reset()

    image = env.render_image()
    x = image_to_tensor(image).to(device)
    with torch.no_grad():
        state = vision_model(x)[0].tolist()

    total_reward = 0
    done = False
    step = 0

    print("\n" + "=" * 52)
    print("  TEST: SJGVision → SJGAgent (End-to-End)")
    print("=" * 52)
    env.render()

    agent_pos, goal_pos, obs_pos = state_to_readable(state)
    print(
        f"  Visión ve:  Agente {agent_pos}  |  "
        f"Objetivo {goal_pos}  |  Obstáculo {obs_pos}"
    )

    while not done:
        step += 1
        action = agent.act(state, epsilon=0.0)
        _, reward, done = env.step(action)

        next_image = env.render_image()
        next_x = image_to_tensor(next_image).to(device)
        with torch.no_grad():
            next_state = vision_model(next_x)[0].tolist()

        state = next_state
        total_reward += reward

        print(f"\n  Paso {step:2d} | Acción: {ACTIONS[action]} | Reward: {reward:+.1f}")
        env.render()

        agent_pos, goal_pos, obs_pos = state_to_readable(state)
        print(
            f"  Visión ve:  Agente {agent_pos}  |  "
            f"Objetivo {goal_pos}  |  Obstáculo {obs_pos}"
        )

    print("=" * 52)
    result = "¡ÉXITO! Llegó al objetivo 🎯" if total_reward > 0 else "Falló el episodio ❌"
    print(f"  {result}")
    print(f"  Reward total: {total_reward:.2f} | Pasos: {step}")
    print("=" * 52 + "\n")


if __name__ == "__main__":
    main()
