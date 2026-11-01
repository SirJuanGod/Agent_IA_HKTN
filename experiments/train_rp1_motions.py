import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from rbj.environment.rp1_mujoco import RP1MotionEnv


CHECKPOINTS = Path(__file__).resolve().parent.parent / "checkpoints"


def main():
    parser = argparse.ArgumentParser(description="Entrena RP1 para imitar los movimientos de config/motions.json")
    parser.add_argument("--phase", choices=["fixed", "floating"], default="fixed")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    # cada subproceso carga torch (~1 GB); 8 agotan la memoria en una máquina de 12 GB
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--init", help="checkpoint .zip para continuar (p. ej. el de la fase fixed)")
    args = parser.parse_args()

    venv = make_vec_env(
        RP1MotionEnv,
        n_envs=args.envs,
        env_kwargs={"fixed_base": args.phase == "fixed"},
        vec_env_cls=SubprocVecEnv,
    )

    if args.init:
        init = Path(args.init)
        vecnorm = init.with_name(init.stem + "_vecnorm.pkl")

        if not vecnorm.exists():  # copia periódica de SB3: rp1_<n>_steps.zip → rp1_vecnormalize_<n>_steps.pkl
            vecnorm = init.with_name(init.stem.replace("rp1_", "rp1_vecnormalize_", 1) + ".pkl")

        venv = VecNormalize.load(str(vecnorm), venv)
        model = PPO.load(args.init, env=venv, device="cpu")
    else:
        venv = VecNormalize(venv, clip_obs=10.0)
        model = PPO(
            "MlpPolicy",
            venv,
            n_steps=1024,
            batch_size=4096,
            n_epochs=5,
            learning_rate=3e-4,
            gamma=0.99,
            # std inicial ~0.14: el residuo arranca cerca de cero (seguir la referencia)
            policy_kwargs={"net_arch": [256, 256], "log_std_init": -2.0},
            device="cpu",
            verbose=1,
        )

    # guardado periódico: un entrenamiento largo interrumpido no pierde todo
    snapshots = CheckpointCallback(
        save_freq=max(500_000 // args.envs, 1),
        save_path=str(CHECKPOINTS / f"rp1_{args.phase}_snapshots"),
        name_prefix="rp1",
        save_vecnormalize=True,
    )
    model.learn(total_timesteps=args.timesteps, reset_num_timesteps=not args.init, callback=snapshots)

    CHECKPOINTS.mkdir(exist_ok=True)
    out = CHECKPOINTS / f"rp1_motions_{args.phase}.zip"
    model.save(out)
    venv.save(str(out).replace(".zip", "_vecnorm.pkl"))
    print(f"Guardado {out}")


if __name__ == "__main__":
    main()
