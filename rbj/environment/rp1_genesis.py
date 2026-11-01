"""
RP1 imitando keyframes en Genesis-Forge (GPU, miles de entornos en paralelo).

Reproduce el contrato de RP1MotionEnv (MuJoCo) para que la política entrenada aquí
se ejecute sin cambios en el entorno MuJoCo local:
    - observación de 102 valores en el mismo orden (articulaciones en orden MuJoCo)
    - acción = residuo 0.25 sobre la referencia, control PD de posición
    - mismos kp/kv/torque/armature/damping/frictionloss y mismo dt (0.02 s)
La referencia (ángulos, orientación y altura de la base) se precalcula con MuJoCo,
así ambos simuladores persiguen exactamente los mismos objetivos.
"""

import math
import re

import genesis as gs
import numpy as np
import torch
from genesis_forge import ManagedEnvironment
from genesis_forge.managers import (
    ActuatorManager,
    ContactManager,
    EntityManager,
    ObservationManager,
    PositionActionManager,
    RewardManager,
    TerminationManager,
)
from genesis_forge.mdp import terminations

from rbj.environment.rp1_mujoco import ACTION_SCALE, DECIMATION, ROOT, RP1MotionEnv


ROBOT_MJCF = ROOT / "third_party" / "RP1_description" / "mjcf" / "robot.xml"
EPISODE_SECONDS = 6.0

# todo lo que no sea muslo, rodilla o pie tocando el suelo = caída (igual que en MuJoCo)
NON_LEG_LINKS = [
    "base_link", ".*_hip_pitch_link", ".*_hip_roll_link", "waist_.*",
    ".*_shoulder_.*", ".*_elbow_link", ".*_wrist_.*",
]


def bake_reference(dt):
    """
    Precalcula con MuJoCo, en una rejilla de paso dt, la referencia de cada movimiento:
    ángulos (q, qd), orientación y altura de la base tras apoyar el pie (como en el reset
    de RP1MotionEnv) y la gravedad proyectada en la base. También devuelve los parámetros
    de las articulaciones del modelo MuJoCo, que son la fuente de verdad del PD.
    """

    env = RP1MotionEnv(fixed_base=False)
    library, model = env.library, env.model
    frames = [round(duration / dt) for duration in library.durations]

    for duration, n in zip(library.durations, frames):
        assert abs(n * dt - duration) < 1e-6, f"la duración {duration} s no es múltiplo de dt={dt}"

    shape = (len(frames), max(frames))
    tables = {
        "q": np.zeros(shape + (model.nu,)), "qd": np.zeros(shape + (model.nu,)),
        "quat": np.zeros(shape + (4,)), "z": np.zeros(shape), "g": np.zeros(shape + (3,)),
    }

    for m, name in enumerate(library.names):
        for k in range(frames[m]):
            env.reset(options={"motion": name, "phase": k * dt})
            tables["q"][m, k], tables["qd"][m, k] = library.ref(m, k * dt)
            tables["quat"][m, k] = env.data.qpos[3:7]
            tables["z"][m, k] = env.data.qpos[2]
            tables["g"][m, k] = env._base_gravity()

    joints = {
        "names": env.joint_names,
        "kp": model.actuator_gainprm[:, 0],
        "kv": -model.actuator_biasprm[:, 2],
        "max_force": model.actuator_forcerange[:, 1],
        "armature": model.dof_armature[env.vadr],
        "damping": model.dof_damping[env.vadr],
        "frictionloss": model.dof_frictionloss[env.vadr],
    }

    return tables, np.array(frames), library.names, library.phases, joints


class MotionState:
    """
    Movimiento y fotograma de referencia de cada entorno (tensores en GPU).
    """

    def __init__(self, tables, frames, allowed, num_envs):
        to = lambda a: torch.as_tensor(a, dtype=gs.tc_float, device=gs.device)
        self.q, self.qd, self.quat, self.z, self.g = (to(tables[key]) for key in ("q", "qd", "quat", "z", "g"))
        self.frames = torch.as_tensor(frames, device=gs.device)
        self.allowed = torch.as_tensor(allowed, device=gs.device)
        self.m = torch.zeros(num_envs, dtype=torch.long, device=gs.device)
        self.k = torch.zeros(num_envs, dtype=torch.long, device=gs.device)

    def at(self, table, offset=0):
        return table[self.m, (self.k + offset) % self.frames[self.m]]

    def phase(self, offset=0):
        n = self.frames[self.m]
        return 2 * math.pi * ((self.k + offset) % n) / n

    def sample(self, envs_idx):
        choice = torch.randint(len(self.allowed), (len(envs_idx),), device=gs.device)
        self.m[envs_idx] = self.allowed[choice]
        self.k[envs_idx] = (torch.rand(len(envs_idx), device=gs.device) * self.frames[self.m[envs_idx]]).long()


class ReferenceResidualAction(PositionActionManager):
    """
    q_objetivo = q_ref + 0.25 * a, con a en orden MuJoCo (el de la política).
    """

    def process_actions(self, actions):
        env = self.env
        target = torch.empty_like(actions)
        target[:, env.perm] = env.motion.at(env.motion.q) + ACTION_SCALE * actions.clamp(-1.0, 1.0)
        return target.clamp(self._clip_values[:, 0], self._clip_values[:, 1])


def reset_to_reference(env, envs_idx, **_):
    """
    Reference state init: movimiento y fase al azar, pose y base de la referencia.
    """

    motion = env.motion
    motion.sample(envs_idx)
    m, k = motion.m[envs_idx], motion.k[envs_idx]

    q = torch.empty((len(envs_idx), env.num_dof), dtype=gs.tc_float, device=gs.device)
    qd = torch.empty_like(q)
    q[:, env.perm], qd[:, env.perm] = motion.q[m, k], motion.qd[m, k]

    position = torch.zeros((len(envs_idx), 3), dtype=gs.tc_float, device=gs.device)
    position[:, 2] = motion.z[m, k] + 0.002

    env.robot.set_dofs_position(q, env.actuators.dofs_idx, envs_idx=envs_idx, zero_velocity=True)
    env.robot.set_pos(position, envs_idx=envs_idx)
    env.robot.set_quat(motion.quat[m, k], envs_idx=envs_idx)
    env.robot.set_dofs_velocity(qd, env.actuators.dofs_idx, envs_idx=envs_idx)


# --- observaciones (mismo orden que RP1MotionEnv._obs) -------------------------------

def obs_q(env):
    return env.actuators.get_dofs_position()[:, env.perm]


def obs_qd(env):
    return env.actuators.get_dofs_velocity()[:, env.perm]


def obs_q_next(env):
    return env.motion.at(env.motion.q, 1)


def obs_phase(env):
    phase = env.motion.phase()
    return torch.stack([torch.sin(phase), torch.cos(phase)], dim=1)


def obs_motion(env):
    return torch.nn.functional.one_hot(env.motion.m, len(env.motion_names)).to(gs.tc_float)


def obs_gravity(env):
    return env.entity.get_projected_gravity()


def obs_angular_velocity(env):
    return env.entity.get_angular_velocity()


def obs_height(env):
    return env.entity.base_pos[:, 2:3]


# --- recompensas (por paso; el RewardManager multiplica por dt, de ahí el /dt) --------

def reward_pose(env):
    return torch.exp(-2.0 * torch.sum((obs_q(env) - env.motion.at(env.motion.q)) ** 2, dim=1))


def reward_velocity(env):
    return torch.exp(-0.1 * torch.sum((obs_qd(env) - env.motion.at(env.motion.qd)) ** 2, dim=1))


def reward_smooth(env):
    change = env.actions.clamp(-1.0, 1.0) - env.last_actions.clamp(-1.0, 1.0)
    return torch.exp(-0.5 * torch.sum(change ** 2, dim=1))


def reward_orientation(env):
    """
    Inclinación del torso de la referencia: es lo que faltaba en MuJoCo para no volcar.
    """

    return torch.exp(-5.0 * torch.sum((obs_gravity(env) - env.motion.at(env.motion.g)) ** 2, dim=1))


def reward_height(env):
    return torch.exp(-50.0 * (env.entity.base_pos[:, 2] - env.motion.at(env.motion.z)) ** 2)


def tilted(env):
    # bad_orientation de Forge usa |g_xy| y no detecta al robot boca abajo
    return obs_gravity(env)[:, 2] > -0.5


class RP1GenesisEnv(ManagedEnvironment):
    def __init__(self, num_envs=4096, motions=None, headless=True):
        dt = RP1MotionEnv(fixed_base=False).dt
        tables, frames, names, phases, joints = bake_reference(dt)

        super().__init__(num_envs=num_envs, dt=dt, max_episode_length_sec=EPISODE_SECONDS)

        self.motion_names = names
        self.joints = joints
        self._perm = None
        allowed = [names.index(n) for n in motions] if motions else list(range(len(names)))
        self.motion = MotionState(tables, frames, allowed, num_envs)

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=dt, substeps=DECIMATION),
            # el MJCF usa condim 4 (fricción torsional), como MuJoCo
            rigid_options=gs.options.RigidOptions(enable_torsional_friction=True),
            show_viewer=not headless,
        )
        self.terrain = self.scene.add_entity(gs.morphs.Plane())
        self.robot = self.scene.add_entity(gs.morphs.MJCF(file=str(ROBOT_MJCF)))

    def config(self):
        joints = self.joints
        exact = lambda values: {f"^{re.escape(n)}$": float(v) for n, v in zip(joints["names"], values)}

        self.actuators = ActuatorManager(
            self,
            joint_names=".*",
            default_pos=0.0,
            kp=exact(joints["kp"]),
            kv=exact(joints["kv"]),
            max_force=exact(joints["max_force"]),
            armature=exact(joints["armature"]),
            damping=exact(joints["damping"]),
            frictionloss=exact(joints["frictionloss"]),
        )
        ReferenceResidualAction(self, actuator_manager=self.actuators, use_default_offset=False)

        self.entity = EntityManager(self, self.robot, on_reset={"reference": {"fn": reset_to_reference}})
        self.non_leg_contact = ContactManager(
            self, entity=self.robot, link_names=NON_LEG_LINKS, with_entity=self.terrain
        )

        ObservationManager(self, cfg={
            "q": {"fn": obs_q},
            "qd": {"fn": obs_qd},
            "q_next": {"fn": obs_q_next},
            "phase": {"fn": obs_phase},
            "motion": {"fn": obs_motion},
            "gravity": {"fn": obs_gravity},
            "angular_velocity": {"fn": obs_angular_velocity},
            "height": {"fn": obs_height},
        })

        per_step = 1.0 / self.dt
        RewardManager(self, cfg={
            "pose": {"fn": reward_pose, "weight": 0.55 * per_step},
            "velocity": {"fn": reward_velocity, "weight": 0.15 * per_step},
            "smooth": {"fn": reward_smooth, "weight": 0.1 * per_step},
            "orientation": {"fn": reward_orientation, "weight": 0.1 * per_step},
            "height": {"fn": reward_height, "weight": 0.1 * per_step},
        })

        TerminationManager(self, term_cfg={
            "tilted": {"fn": tilted},
            "non_leg_contact": {"fn": terminations.contact_force(contact_manager=self.non_leg_contact)},
            "timeout": {"fn": terminations.timeout(), "time_out": True},
        })

    @property
    def perm(self):
        """
        perm[j] = índice Genesis de la articulación j en orden MuJoCo (la política usa ese orden).
        """

        if self._perm is None:
            genesis_order = list(self.actuators.dofs_names)
            assert sorted(genesis_order) == sorted(self.joints["names"]), genesis_order
            self._perm = torch.tensor([genesis_order.index(n) for n in self.joints["names"]], device=gs.device)

        return self._perm

    @property
    def num_dof(self):
        return len(self.joints["names"])

    def step(self, actions):
        # mismo calendario que MuJoCo: objetivo y recompensa con ref(t+dt), observación con ref(t+2dt)
        self.motion.k += 1
        return super().step(actions)
