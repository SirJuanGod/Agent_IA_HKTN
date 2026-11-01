import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import mujoco
import numpy as np

from video_to_motion import BONES, FEET, UNOBSERVED, Retargeter, to_keyframes


def robot_landmarks(retargeter, q):
    """
    Landmarks "humanos" sacados del propio RP1 en la pose q.
    """

    m, d = retargeter.model, retargeter.data
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)

    points = np.zeros((33, 3))

    for a, b, body_a, body_b in BONES:
        points[a] = d.xpos[m.body(body_a).id]
        points[b] = d.xpos[m.body(body_b).id]

    for heel, toe, foot in FEET:
        body = m.body(foot).id
        axis = d.xmat[body].reshape(3, 3)[:, 0]
        points[heel] = d.xpos[body] - 0.05 * axis
        points[toe] = d.xpos[body] + 0.10 * axis

    return points


def main():
    rng = np.random.default_rng(0)
    retargeter = Retargeter()
    low, high = retargeter.low, retargeter.high

    # rango moderado: desde q=0 la IK puede caer en mínimos locales con poses extremas
    # (en un video cada frame parte de la solución del anterior)
    q_true = rng.uniform(0.3 * low, 0.3 * high)

    # piernas en sentadilla con los pies planos (el retargeting asume la planta apoyada)
    legs = {"hip_pitch": (-0.6, 0.6), "hip_roll": (0, 0), "hip_yaw": (0, 0),
            "knee": (-1.2, 1.2), "ankle_pitch": (-0.6, -0.6), "ankle_roll": (0, 0)}

    for joint, (left, right) in legs.items():
        q_true[retargeter.model.joint(f"left_{joint}_joint").qposadr[0]] = left
        q_true[retargeter.model.joint(f"right_{joint}_joint").qposadr[0]] = right

    for name in UNOBSERVED:
        q_true[retargeter.model.joint(name).qposadr[0]] = 0.0

    points = robot_landmarks(retargeter, q_true)

    # ejes de MediaPipe (x derecha imagen, y abajo, z profundidad) + escala humana
    to_mediapipe = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
    points = 1.7 * points @ to_mediapipe.T + np.array([0.3, -0.2, 2.0])

    retargeter.q[:] = 0.0
    q, error = retargeter.solve(points, iterations=200)

    print(f"error de dirección: {error:.2e}")
    assert error < 1e-3, error

    keyframes, duration = to_keyframes(np.array([0.0, 1.0]), np.stack([q, q]), ["a"] * len(q), 10, 0.5)
    assert keyframes[0] == {"t": 0.0, "joints": {}} and duration == 2.0, (keyframes[:2], duration)

    print("test_video_to_motion OK")


if __name__ == "__main__":
    main()
