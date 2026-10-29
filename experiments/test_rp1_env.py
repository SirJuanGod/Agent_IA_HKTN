import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np

from rbj.environment.motion_library import MotionLibrary
from rbj.environment.rp1_mujoco import MOTIONS_PATH, RP1MotionEnv


def main():
    fixed = RP1MotionEnv(fixed_base=True)
    floating = RP1MotionEnv(fixed_base=False)

    assert fixed.model.nu == 28, fixed.model.nu
    assert fixed.observation_space.shape == floating.observation_space.shape
    assert set(fixed.motions) < set(floating.motions)

    library = fixed.library

    for i, name in enumerate(library.names):
        for k, t in enumerate(library._times[i]):
            q, _ = library.ref(i, t)
            assert np.allclose(q, library._poses[i][k]), (name, t)

    try:
        MotionLibrary(MOTIONS_PATH, ["solo_una_articulacion"])
        raise AssertionError("debería rechazar articulaciones desconocidas")
    except ValueError:
        pass

    for env in (fixed, floating):
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs)

        for _ in range(100):
            obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            assert np.all(np.isfinite(obs)) and 0.0 <= reward <= 1.0

            if terminated or truncated:
                obs, _ = env.reset()

    # con acción 0 y base fija la política "perfecta" es seguir la referencia
    obs, _ = fixed.reset(options={"motion": "wave", "phase": 0.0})
    pose = [fixed.step(np.zeros(28))[4]["pose_reward"] for _ in range(50)]
    print(f"pose_reward medio con acción cero (wave): {np.mean(pose):.3f}")

    print("test_rp1_env OK")


if __name__ == "__main__":
    main()
