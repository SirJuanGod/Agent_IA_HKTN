import argparse
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import mujoco
import mujoco.viewer

from rbj.environment.rp1_mujoco import RP1Environment, RP1MotionEnv


CHECKPOINTS = Path(__file__).resolve().parent.parent / "checkpoints"


def play_reference(motion, fixed_base):
    """
    Reproduce los keyframes de forma cinemática (sin física ni política)
    para diseñar y ajustar config/motions.json.
    """

    env = RP1MotionEnv(fixed_base=fixed_base)
    index = env.library.index(motion)
    t = 0.0

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            q, _ = env.library.ref(index, t)
            env.data.qpos[env.qadr] = q
            mujoco.mj_forward(env.model, env.data)
            viewer.sync()
            time.sleep(env.dt)
            t += env.dt


def main():
    parser = argparse.ArgumentParser(description="Visualiza un movimiento de RP1")
    parser.add_argument("--motion", default="wave")
    parser.add_argument("--phase", choices=["fixed", "floating"], default="fixed")
    parser.add_argument("--reference", action="store_true", help="solo keyframes, sin política")
    parser.add_argument("--policy", help="política .pt de Genesis-Forge (por defecto, el .zip de SB3 de la fase)")
    args = parser.parse_args()

    fixed_base = args.phase == "fixed"

    if args.reference:
        play_reference(args.motion, fixed_base)
        return

    policy = args.policy or CHECKPOINTS / f"rp1_motions_{args.phase}.zip"
    robot = RP1Environment(policy, str(policy).replace(".zip", "_vecnorm.pkl"), fixed_base, viewer=True)
    robot.reset()

    while robot.viewer.is_running() and not robot.is_done():
        print(robot.step({"action": "PERFORM", "motion": args.motion}))

    robot.close()


if __name__ == "__main__":
    main()
