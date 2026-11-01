"""
Evalúa en MuJoCo los checkpoints de RSL-RL entrenados en Genesis y exporta el mejor.

El resultado oscila entre checkpoints y el último no tiene por qué ser el que mejor se
transfiere a MuJoCo, así que se elige midiendo en MuJoCo:

    python experiments/select_rp1_policy.py logs/rp1_genesis_colab --first 1000
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from rbj.environment.rp1_mujoco import RP1MotionEnv
from train_rp1_genesis import TRAIN_CFG


ROOT = Path(__file__).resolve().parent.parent
START_PHASES = (0.0, 1 / 3, 2 / 3)  # fracción del movimiento
MAX_STEPS = 400  # 8 s


def load_actor(path, obs_size, num_actions):
    cfg = {k: v for k, v in TRAIN_CFG["actor"].items() if k != "class_name"}
    obs = TensorDict({"policy": torch.zeros(1, obs_size)}, batch_size=[1])
    actor = MLPModel(obs, TRAIN_CFG["obs_groups"], "actor", num_actions, **cfg)
    actor.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["actor_state_dict"])
    return torch.jit.script(actor.as_jit())


def evaluate(policy, env):
    """
    Por movimiento: fracción de pasos sobrevividos y pose media, promediando las fases de inicio.
    """

    results = {}

    for index in env.motions:
        name = env.library.names[index]
        survival, pose = [], []

        for start in START_PHASES:
            obs, _ = env.reset(options={"motion": name, "phase": start * env.library.durations[index]})
            rewards = []

            for step in range(MAX_STEPS):
                action = policy(torch.as_tensor(obs)[None]).detach().numpy()[0]
                obs, _, terminated, _, info = env.step(action)
                rewards.append(info["pose_reward"])

                if terminated:
                    break

            survival.append(len(rewards) / MAX_STEPS)
            pose.append(np.mean(rewards))

        results[name] = (float(np.mean(survival)), float(np.mean(pose)))

    return results


def main():
    parser = argparse.ArgumentParser(description="Elige el checkpoint que mejor funciona en MuJoCo")
    parser.add_argument("log_dir")
    parser.add_argument("--first", type=int, default=0, help="ignora checkpoints anteriores a esta iteración")
    parser.add_argument("--every", type=int, default=100)
    parser.add_argument("--output", default=str(ROOT / "checkpoints" / "rp1_genesis_policy.pt"))
    args = parser.parse_args()

    env = RP1MotionEnv(fixed_base=False)
    obs_size, num_actions = env.observation_space.shape[0], env.action_space.shape[0]
    iteration = lambda path: int(re.findall(r"model_(\d+)", path.name)[0])
    paths = sorted(Path(args.log_dir).glob("model_*.pt"), key=iteration)
    paths = [p for p in paths if iteration(p) >= args.first and iteration(p) % args.every == 0]

    if not paths:
        raise ValueError(f"No hay checkpoints en {args.log_dir} desde la iteración {args.first}")

    best = None

    for path in paths:
        policy = load_actor(path, obs_size, num_actions)
        results = evaluate(policy, env)
        survival = np.mean([s for s, _ in results.values()])
        pose = np.mean([p for _, p in results.values()])
        full = sum(s >= 0.999 for s, _ in results.values())
        print(f"{path.name:16} aguantan {full}/{len(results)} | supervivencia {survival:.3f} | pose {pose:.3f}", flush=True)

        # sobrevivir manda; la pose desempata entre políticas igual de estables
        score = survival + 0.1 * pose

        if best is None or score > best[0]:
            best = (score, path, policy, results)

    _, path, policy, results = best
    policy.save(args.output)

    print(f"\nMejor: {path.name} → {args.output}")
    for name, (survival, pose) in results.items():
        print(f"  {name:18} supervivencia {survival:.2f} | pose {pose:.3f}")


if __name__ == "__main__":
    main()
