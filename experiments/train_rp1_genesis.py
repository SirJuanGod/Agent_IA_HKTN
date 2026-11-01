"""
Entrena RP1 en Genesis-Forge con RSL-RL (pensado para GPU, p. ej. Google Colab) y exporta
la política a TorchScript para ejecutarla en el entorno MuJoCo local:

    python experiments/train_rp1_genesis.py --envs 4096 --iterations 3000 --log-dir /content/drive/MyDrive/rp1
    python experiments/run_rp1_language.py --phase floating --policy <log-dir>/rp1_policy.pt
"""

import argparse
import copy
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import genesis as gs
import torch


TRAIN_CFG = {
    "algorithm": {
        "class_name": "PPO", "clip_param": 0.2, "desired_kl": 0.01, "entropy_coef": 0.005,
        "gamma": 0.99, "lam": 0.95, "learning_rate": 1e-3, "max_grad_norm": 1.0,
        "num_learning_epochs": 5, "num_mini_batches": 4, "schedule": "adaptive", "value_loss_coef": 1.0,
    },
    "actor": {
        "class_name": "MLPModel", "hidden_dims": [512, 256, 128], "activation": "elu",
        "obs_normalization": True,
        # residuo pequeño al principio: la referencia sola ya es una buena primera aproximación
        "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.3},
    },
    "critic": {
        "class_name": "MLPModel", "hidden_dims": [512, 256, 128], "activation": "elu",
        "obs_normalization": True,
    },
    "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
    "num_steps_per_env": 24,
    "save_interval": 100,
    "seed": 1,
}


def main():
    parser = argparse.ArgumentParser(description="Entrena RP1 (9 movimientos) en Genesis-Forge con RSL-RL")
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--log-dir", default=str(Path(__file__).resolve().parent.parent / "logs" / "rp1_genesis"))
    parser.add_argument("--resume", help="checkpoint model_<it>.pt de RSL-RL para continuar")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu, logging_level="warning")

    from rbj.environment.rp1_genesis import RP1GenesisEnv

    genesis_env = RP1GenesisEnv(num_envs=args.envs)
    # construir ANTES de importar RSL-RL: arrastra TensorBoard → TensorFlow (preinstalado en
    # Colab), y con TensorFlow cargado el contexto EGL de Genesis hace segfault en scene.build()
    genesis_env.build()

    from genesis_forge.wrappers import RslRlWrapper
    from rsl_rl.runners import OnPolicyRunner

    env = RslRlWrapper(genesis_env)
    env.reset()

    runner = OnPolicyRunner(env, copy.deepcopy(TRAIN_CFG), args.log_dir, device=gs.device)

    if args.resume:
        runner.load(args.resume)

    # init_at_random_ep_len de RSL-RL no funciona con Forge (max_episode_length es por entorno):
    # se reparte a mano para que los episodios no terminen todos a la vez
    episode = env.unwrapped.episode_length
    episode[:] = torch.randint_like(episode, high=int(env.unwrapped.max_episode_length.max()))

    runner.learn(num_learning_iterations=args.iterations)

    runner.export_policy_to_jit(args.log_dir, "rp1_policy.pt")
    print(f"Política exportada: {Path(args.log_dir) / 'rp1_policy.pt'}", flush=True)

    # al destruir el contexto EGL de Genesis el intérprete hace segfault al salir (Colab):
    # todo está guardado, así que se sale sin ese teardown
    os._exit(0)


if __name__ == "__main__":
    main()
