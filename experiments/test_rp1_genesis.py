"""
Paridad Genesis-Forge ↔ MuJoCo: misma observación al arrancar el mismo fotograma y
deriva de ambos simuladores con acción cero. Corre en CPU (no necesita GPU).
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import genesis as gs
import numpy as np
import torch

from rbj.environment.rp1_genesis import RP1GenesisEnv
from rbj.environment.rp1_mujoco import RP1MotionEnv


LAYOUT = [("q", 28), ("qd", 28), ("q_next", 28), ("phase", 2), ("motion", 9), ("gravity", 3), ("angvel", 3), ("height", 1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", default="squat")
    parser.add_argument("--frame", type=int, default=30)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    gs.init(backend=gs.cpu, logging_level="warning")
    genesis = RP1GenesisEnv(num_envs=1, motions=[args.motion], randomize=False)
    genesis.build()

    # el reset de referencia arranca siempre en el mismo fotograma que MuJoCo
    motion_idx = genesis.motion_names.index(args.motion)

    def fixed_frame(envs_idx):
        genesis.motion.m[envs_idx] = motion_idx
        genesis.motion.k[envs_idx] = args.frame

    genesis.motion.sample = fixed_frame
    obs_genesis, _ = genesis.reset()
    obs_genesis = obs_genesis[0].cpu().numpy()

    mujoco = RP1MotionEnv(fixed_base=False)
    obs_mujoco, _ = mujoco.reset(options={"motion": args.motion, "phase": args.frame * mujoco.dt})

    assert obs_genesis.shape == obs_mujoco.shape, (obs_genesis.shape, obs_mujoco.shape)

    start = 0
    for name, size in LAYOUT:
        diff = np.abs(obs_genesis[start:start + size] - obs_mujoco[start:start + size]).max()
        print(f"{name:8} max |genesis - mujoco| = {diff:.4f}")
        # la altura lleva 2 mm de margen a propósito; lo demás debe ser idéntico
        assert diff < (0.005 if name == "height" else 1e-3), f"observación '{name}' distinta en Genesis y MuJoCo"
        start += size

    zero = torch.zeros((1, 28), device=gs.device)
    for step in range(1, args.steps + 1):
        obs_g, _, terminated_g, _, _ = genesis.step(zero)
        obs_m, _, terminated_m, _, _ = mujoco.step(np.zeros(28))

        if step % 10 == 0 or terminated_g.any() or terminated_m:
            obs_g = obs_g[0].cpu().numpy()
            start, diffs = 0, []
            for name, size in LAYOUT:
                diffs.append(f"{name} {np.abs(obs_g[start:start + size] - obs_m[start:start + size]).max():.3f}")
                start += size
            print(f"paso {step:3}: " + " | ".join(diffs) +
                  f" | terminado g/m {bool(terminated_g.any())}/{terminated_m}")

        if terminated_g.any() or terminated_m:
            break


if __name__ == "__main__":
    main()
