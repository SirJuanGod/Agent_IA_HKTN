import time
from pathlib import Path
from typing import Any, Dict

import gymnasium as gym
import mujoco
import numpy as np

from rbj.environment.base import Environment
from rbj.environment.motion_library import MotionLibrary


ROOT = Path(__file__).resolve().parents[2]
MJCF_PATH = ROOT / "third_party" / "RP1_description" / "mjcf" / "rp1_flat.xml"
MOTIONS_PATH = ROOT / "config" / "motions.json"

FIXED_BASE_HEIGHT = 0.9
DECIMATION = 4  # timestep del MJCF = 0.005 s → control a 50 Hz
ACTION_SCALE = 0.25


def build_model(fixed_base):
    """
    Compila el MJCF de RP1 cambiando los motores de torque por actuadores
    de posición (PD implícito de MuJoCo) con el mismo límite de torque.
    """

    spec = mujoco.MjSpec.from_file(str(MJCF_PATH))

    if fixed_base:
        spec.delete(spec.joint("base_link_freejoint"))
        spec.body("base_link").pos = [0, 0, FIXED_BASE_HEIGHT]

    for actuator in spec.actuators:
        torque = np.array(actuator.ctrlrange)

        actuator.set_to_position(kp=1.5 * torque[1], kv=0.05 * torque[1])
        actuator.forcerange = torque
        actuator.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        actuator.ctrlrange = spec.joint(actuator.target).range
        actuator.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE

    return spec.compile()


class RP1MotionEnv(gym.Env):
    """
    Entorno Gymnasium: RP1 imita movimientos definidos por keyframes.

    Una sola política para todos los movimientos: la observación lleva
    el one-hot del movimiento y la acción es un residuo sobre la referencia.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        fixed_base=True,
        motions=None,
        motions_path=MOTIONS_PATH,
        episode_seconds=6.0,
    ):
        self.fixed_base = fixed_base
        self.model = build_model(fixed_base)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep * DECIMATION

        self.joint_names = [
            self.model.joint(self.model.actuator_trnid[i, 0]).name
            for i in range(self.model.nu)
        ]
        joint_ids = self.model.actuator_trnid[:, 0]
        self.qadr = self.model.jnt_qposadr[joint_ids]
        self.vadr = self.model.jnt_dofadr[joint_ids]
        self.ctrl_range = self.model.actuator_ctrlrange.copy()

        self.library = MotionLibrary(motions_path, self.joint_names, self.ctrl_range)

        if motions is None:
            motions = [
                name for name, phase in zip(self.library.names, self.library.phases)
                if not fixed_base or phase == "fixed"
            ]

        self.motions = [self.library.index(name) for name in motions]
        self.max_steps = int(episode_seconds / self.dt)
        self.base_id = self.model.body("base_link").id

        n = self.model.nu
        obs_size = 3 * n + 2 + len(self.library.names) + 7

        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_size,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (n,), np.float32)

        self.motion = self.motions[0]
        self.t = 0.0
        self.steps = 0
        self.prev_action = np.zeros(n)

    def start_motion(self, name, phase=0.0):
        """
        Cambia el movimiento objetivo sin tocar el estado físico.
        """

        self.motion = self.library.index(name)
        self.t = phase
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}

        if "motion" in options:
            self.motion = self.library.index(options["motion"])
        else:
            self.motion = self.motions[self.np_random.integers(len(self.motions))]

        duration = self.library.durations[self.motion]
        self.t = options.get("phase", self.np_random.uniform(0, duration))
        self.steps = 0
        self.prev_action = np.zeros(self.model.nu)

        # reference state init: arranca en la pose de referencia de esa fase
        mujoco.mj_resetData(self.model, self.data)
        q_ref, qd_ref = self.library.ref(self.motion, self.t)
        self.data.qpos[self.qadr] = q_ref
        self.data.qvel[self.vadr] = qd_ref
        self.data.ctrl[:] = q_ref

        if not self.fixed_base:
            mujoco.mj_kinematics(self.model, self.data)
            self.data.qpos[2] -= self._lowest_geom_z() - 0.001

        mujoco.mj_forward(self.model, self.data)

        return self._obs(), {"motion": self.library.names[self.motion]}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        q_ref, _ = self.library.ref(self.motion, self.t + self.dt)
        target = q_ref + ACTION_SCALE * action
        self.data.ctrl[:] = np.clip(target, self.ctrl_range[:, 0], self.ctrl_range[:, 1])

        mujoco.mj_step(self.model, self.data, nstep=DECIMATION)

        self.t += self.dt
        self.steps += 1

        q_ref, qd_ref = self.library.ref(self.motion, self.t)
        q = self.data.qpos[self.qadr]
        qd = self.data.qvel[self.vadr]

        pose = np.exp(-2.0 * np.sum((q - q_ref) ** 2))
        vel = np.exp(-0.1 * np.sum((qd - qd_ref) ** 2))
        smooth = np.exp(-0.5 * np.sum((action - self.prev_action) ** 2))
        reward = 0.65 * pose + 0.15 * vel + 0.2 * smooth

        self.prev_action = action

        terminated = False

        if not self.fixed_base:
            gravity_z = self._base_gravity()[2]
            terminated = self.data.xpos[self.base_id][2] < 0.45 or gravity_z > -0.5

        truncated = self.steps >= self.max_steps
        info = {"motion": self.library.names[self.motion], "pose_reward": pose}

        return self._obs(), float(reward), bool(terminated), truncated, info

    def _base_gravity(self):
        rotation = self.data.xmat[self.base_id].reshape(3, 3)
        return rotation.T @ np.array([0.0, 0.0, -1.0])

    def _lowest_geom_z(self):
        collision = self.model.geom_contype > 0
        collision &= self.model.geom_bodyid > 0
        return np.min(self.data.geom_xpos[collision, 2])

    def _obs(self):
        duration = self.library.durations[self.motion]
        phase = 2 * np.pi * (self.t % duration) / duration
        q_next, _ = self.library.ref(self.motion, self.t + self.dt)

        rotation = self.data.xmat[self.base_id].reshape(3, 3)
        angular_velocity = rotation.T @ self.data.cvel[self.base_id][:3]
        one_hot = np.zeros(len(self.library.names))
        one_hot[self.motion] = 1.0

        return np.concatenate([
            self.data.qpos[self.qadr],
            self.data.qvel[self.vadr],
            q_next,
            [np.sin(phase), np.cos(phase)],
            one_hot,
            self._base_gravity(),
            angular_velocity,
            [self.data.xpos[self.base_id][2]],
        ]).astype(np.float32)


class RP1Environment(Environment):
    """
    Adaptador al Environment de SJG-Agent: ejecuta la política entrenada.

    Acciones:
        {"action": "PERFORM", "motion": "wave"}
        {"action": "STOP"}
    """

    def __init__(self, policy_path, vecnorm_path, fixed_base=True, viewer=False):
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        self.env = RP1MotionEnv(fixed_base=fixed_base)
        self.policy = PPO.load(policy_path, device="cpu")
        self.normalizer = VecNormalize.load(vecnorm_path, DummyVecEnv([lambda: self.env]))
        self.normalizer.training = False

        self.viewer = None

        if viewer:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.env.model, self.env.data)

        self.obs = None
        self.done = False
        self.last_motion = None

    def reset(self) -> Dict[str, Any]:
        rest = self.env.library.names[self.env.motions[0]]
        self.obs, _ = self.env.reset(options={"motion": rest, "phase": 0.0})
        self.done = False
        self.last_motion = None
        self._sync()

        return self.observe()

    def observe(self) -> Dict[str, Any]:
        return {
            "environment": "rp1_mujoco",
            "motion": self.last_motion,
            "available_motions": list(self.env.library.names),
            "base_height": float(self.env.data.xpos[self.env.base_id][2]),
            "fallen": self.done,
        }

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if self.obs is None:
            self.reset()

        if action.get("action") == "STOP":
            self.last_motion = None
            return self.observe()

        if action.get("action") != "PERFORM":
            raise ValueError(f"RP1Environment no soporta la acción {action}")

        motion = action["motion"]

        if motion not in self.env.library.names:
            raise ValueError(f"Movimiento desconocido: {motion}")

        self.env.start_motion(motion)
        self.last_motion = motion
        steps = int(self.env.library.durations[self.env.motion] / self.env.dt)

        for _ in range(steps):
            norm_obs = self.normalizer.normalize_obs(self.obs)
            act, _ = self.policy.predict(norm_obs, deterministic=True)
            self.obs, _, terminated, _, _ = self.env.step(act)
            self._sync()

            if terminated:
                self.done = True
                break

        return self.observe()

    def is_done(self) -> bool:
        return self.done

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()

    def _sync(self):
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()
            time.sleep(self.env.dt)
