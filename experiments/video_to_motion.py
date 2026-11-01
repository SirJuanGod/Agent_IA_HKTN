import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import mujoco
import numpy as np

from rbj.environment.rp1_mujoco import MOTIONS_PATH, ROOT, build_model


POSE_MODEL = ROOT / "data" / "models" / "pose_landmarker_full.task"
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"

CONTINUITY = 0.5  # peso de |q - q_frame_anterior| en la IK

# MediaPipe world landmarks: y hacia abajo (gravedad si la cámara está nivelada)
MEDIAPIPE_DOWN = (0.0, 1.0, 0.0)

# Landmarks de MediaPipe Pose (izquierda/derecha de la persona)
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 23, 24, 25, 26, 27, 28
L_HEEL, R_HEEL, L_TOE, R_TOE = 29, 30, 31, 32

# hueso: (landmark origen, landmark destino, cuerpo RP1 origen, cuerpo RP1 destino)
BONES = []
for side, s in (("left", 0), ("right", 1)):
    BONES += [
        (L_SHOULDER + s, L_ELBOW + s, f"{side}_shoulder_roll_link", f"{side}_elbow_link"),
        (L_ELBOW + s, L_WRIST + s, f"{side}_elbow_link", f"{side}_wrist_pitch_link"),
        (L_HIP + s, L_KNEE + s, f"{side}_hip_roll_link", f"{side}_knee_link"),
        (L_KNEE + s, L_ANKLE + s, f"{side}_knee_link", f"{side}_ankle_pitch_link"),
    ]

# shortcut: con una sola cámara la profundidad de los hombros salta (giros de cintura de
# ±170° entre frames con los brazos arriba), así que el waist yaw no se toma del video;
# para movimientos con torsión haría falta 3D multi-cámara o un suavizado más fuerte

# pie: dirección talón→punta contra el eje x del pie del robot
FEET = [(L_HEEL, L_TOE, "left_ankle_roll_link"), (R_HEEL, R_TOE, "right_ankle_roll_link")]

# sin información en el video: se mantienen cerca de 0
UNOBSERVED = ["waist_roll_joint", "waist_yaw_joint"] + [
    f"{side}_wrist_{axis}_joint" for side in ("left", "right") for axis in ("roll", "yaw", "pitch")
]


def unit(v):
    return v / np.linalg.norm(v)


def human_frame(points, down):
    """
    Expresa los landmarks y la gravedad (`down`, en los ejes de entrada)
    en el marco del RP1 (x adelante, y izquierda, z arriba).

    El "arriba" es cadera→hombros: el RP1 no tiene pitch de cintura, así que
    torso y pelvis son un mismo sólido.
    """

    hip = (points[L_HIP] + points[R_HIP]) / 2
    left = unit(points[L_HIP] - points[R_HIP])
    up = (points[L_SHOULDER] + points[R_SHOULDER]) / 2 - hip
    up = unit(up - up @ left * left)
    forward = np.cross(left, up)
    rotation = np.stack([forward, left, up])

    return (points - hip) @ rotation.T, rotation @ down


class Retargeter:
    """
    IK por direcciones de huesos (independiente de la escala humano/robot)
    con mínimos cuadrados amortiguados sobre los Jacobianos de MuJoCo.
    """

    def __init__(self):
        self.model = build_model(fixed_base=True)
        self.data = mujoco.MjData(self.model)
        self.low, self.high = self.model.jnt_range[:, 0], self.model.jnt_range[:, 1]
        self.q = np.zeros(self.model.nq)

        body = lambda name: self.model.body(name).id
        self.bones = [(a, b, body(ra), body(rb)) for a, b, ra, rb in BONES]
        self.feet = [(a, b, body(r)) for a, b, r in FEET]
        self.unobserved = [self.model.joint(name).qposadr[0] for name in UNOBSERVED]
        self.previous = None

        # brazo de cada geometría (None si no es brazo) y su shoulder roll hacia fuera
        arm_side = lambda b: next((s for s in ("left", "right") if self.model.body(b).name.startswith(f"{s}_")
                                   and any(p in self.model.body(b).name for p in ("shoulder", "elbow", "wrist"))), None)
        self.geom_arm = [arm_side(self.model.geom_bodyid[g]) for g in range(self.model.ngeom)]
        self.abduct = {s: (self.model.joint(f"{s}_shoulder_roll_joint").qposadr[0], 1.0 if s == "left" else -1.0)
                       for s in ("left", "right")}

    def _residual(self, target, down):
        m, d = self.model, self.data
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)

        errors, jacobians = [], []
        jacp_a, jacp_b, jacr = (np.zeros((3, m.nv)) for _ in range(3))

        for a, b, body_a, body_b in self.bones:
            vector = d.xpos[body_b] - d.xpos[body_a]
            length = np.linalg.norm(vector)
            direction = vector / length

            mujoco.mj_jacBody(m, d, jacp_a, None, body_a)
            mujoco.mj_jacBody(m, d, jacp_b, None, body_b)

            errors.append(direction - unit(target[b] - target[a]))
            jacobians.append((np.eye(3) - np.outer(direction, direction)) / length @ (jacp_b - jacp_a))

        for heel, toe, foot in self.feet:
            rotation = d.xmat[foot].reshape(3, 3)
            mujoco.mj_jacBody(m, d, None, jacr, foot)

            # talón/punta son los landmarks más ruidosos (su giro horizontal salta de un frame
            # a otro): el pie apunta hacia donde mira la pelvis y del video solo se toma cuánto
            # baja la punta. Planta apoyada salvo de puntillas (punta >64° abajo; de pie ya da
            # ~25° porque el landmark del talón está más alto que la punta)
            steep = unit(target[toe] - target[heel]) @ down
            flat = steep < 0.9
            forward = unit(np.array([1.0, 0.0, 0.0]) - down[0] * down)
            pitch = 0.0 if flat else steep
            axes = [(rotation[:, 0], np.sqrt(1 - pitch ** 2) * forward + pitch * down)]

            if flat:  # planta horizontal: fija también el roll del tobillo
                axes.append((rotation[:, 2], -down))

            for axis, wanted in axes:
                errors.append(axis - wanted)
                jacobians.append(-np.cross(axis, np.eye(3)).T @ jacr)

        fitted = len(errors)
        regularization = np.zeros((len(self.unobserved), m.nv))
        regularization[np.arange(len(self.unobserved)), self.unobserved] = 0.3

        errors.append(0.3 * self.data.qpos[self.unobserved])
        jacobians.append(regularization)

        # continuidad: lo que el video no fija (p. ej. shoulder yaw con el brazo recto)
        # se queda donde estaba en vez de saltar de rama entre frames
        if self.previous is not None:
            errors.append(CONTINUITY * (self.data.qpos - self.previous))
            jacobians.append(CONTINUITY * np.eye(m.nv))

        return np.concatenate(errors), np.vstack(jacobians), 3 * fitted

    def _clear_arms(self, step=0.05, attempts=10, tolerance=0.01):
        """
        El RP1 es más ancho que una persona: un brazo pegado al cuerpo en el video
        atraviesa el torso o el muslo del robot. Se abre (shoulder roll) hasta despejarlo,
        como mucho step*attempts rad (hay contactos que abrir el brazo no resuelve).
        """

        for _ in range(attempts):
            self.data.qpos[:] = self.q
            mujoco.mj_fwdPosition(self.model, self.data)
            contact = self.data.contact
            touching = {self.geom_arm[g] for g1, g2, dist in zip(contact.geom1, contact.geom2, contact.dist)
                        if dist < -tolerance for g in (g1, g2)} - {None}

            if not touching:
                return

            for side in touching:
                joint, outward = self.abduct[side]
                self.q[joint] = np.clip(self.q[joint] + outward * step, self.low[joint], self.high[joint])

    def solve(self, landmarks, down=MEDIAPIPE_DOWN, iterations=30, damping=1e-2):
        """
        landmarks: (33, 3) en cualquier sistema de ejes dextrógiro; down: gravedad en esos ejes.
        Arranca desde la solución anterior (continuidad temporal).
        """

        target, down = human_frame(landmarks, np.asarray(down, dtype=float))

        for _ in range(iterations):
            self.data.qpos[:] = self.q
            error, jacobian, _ = self._residual(target, down)
            step = np.linalg.solve(jacobian.T @ jacobian + damping * np.eye(self.model.nv), jacobian.T @ error)
            self.q = np.clip(self.q - step, self.low, self.high)

        self._clear_arms()
        self.data.qpos[:] = self.q
        error, _, fitted = self._residual(target, down)
        self.previous = self.q.copy()

        return self.q.copy(), float(np.mean(error[:fitted] ** 2))


def extract_landmarks(video, start, end):
    """
    Devuelve (tiempos, landmarks 3D) de MediaPipe Pose entre start y end segundos.
    """

    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

    if not POSE_MODEL.exists():
        raise FileNotFoundError(f"Falta {POSE_MODEL}; descárgalo de {POSE_MODEL_URL}")

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=RunningMode.VIDEO,
    )
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)

    if not fps > 0:
        raise ValueError(f"No se pudo leer {video} (fps={fps})")

    times, frames = [], []

    with PoseLandmarker.create_from_options(options) as landmarker:
        index = 0

        while True:
            ok, frame = capture.read()
            t = index / fps
            index += 1

            if not ok or t > end:
                break

            if t < start:
                continue

            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = landmarker.detect_for_video(image, int(t * 1000))

            if result.pose_world_landmarks:
                points = result.pose_world_landmarks[0]
                times.append(t - start)
                frames.append([[p.x, p.y, p.z] for p in points])

    if not frames:
        raise ValueError(f"MediaPipe no detectó ninguna persona en {video} [{start}, {end}] s")

    return np.array(times), np.array(frames)


def to_keyframes(times, poses, joint_names, rate, blend):
    """
    Submuestrea a `rate` Hz y añade la pose de reposo al inicio y al final
    (los demás movimientos de motions.json empiezan y acaban en reposo).
    """

    sample_times = np.arange(0.0, times[-1] + 1e-9, 1.0 / rate)
    keyframes = [{"t": 0.0, "joints": {}}]

    for t in sample_times:
        q = np.array([np.interp(t, times, poses[:, j]) for j in range(poses.shape[1])])
        keyframes.append({
            "t": round(blend + float(t), 3),
            "joints": {name: round(float(v), 3) for name, v in zip(joint_names, q) if abs(v) > 1e-3},
        })

    return keyframes, round(blend + float(sample_times[-1]) + blend, 3)


def main():
    parser = argparse.ArgumentParser(description="Video humano → keyframes de RP1 (MediaPipe Pose + IK)")
    parser.add_argument("video")
    parser.add_argument("--name", required=True, help="nombre del movimiento en motions.json")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=1e9)
    parser.add_argument("--phase", choices=["fixed", "floating"], default="floating")
    parser.add_argument("--rate", type=float, default=10.0, help="keyframes por segundo")
    parser.add_argument("--source", default="", help="atribución del video (título y licencia)")
    parser.add_argument("--slowdown", type=float, default=1.0, help="estira el tiempo (2 = el doble de lento)")
    parser.add_argument("--leg-amplitude", type=float, default=1.0,
                        help="escala las articulaciones de las piernas hacia la postura de pie (0.7 = 70%%)")
    args = parser.parse_args()

    times, landmarks = extract_landmarks(args.video, args.start, args.end)

    # mediana de 3 frames: quita fallos de un solo frame (p. ej. izquierda/derecha intercambiadas)
    padded = np.concatenate([landmarks[:1], landmarks, landmarks[-1:]])
    landmarks = np.median(np.stack([padded[:-2], padded[1:-1], padded[2:]]), axis=0)

    retargeter = Retargeter()
    solutions = [retargeter.solve(frame) for frame in landmarks]
    poses = np.array([q for q, _ in solutions])
    errors = [e for _, e in solutions]

    print(f"{len(times)} frames con pose, error medio de dirección {np.mean(errors):.4f}")

    joint_names = [retargeter.model.joint(j).name for j in range(retargeter.model.njnt)]

    # suavizado para el robot: la forma humana se conserva, la dificultad dinámica baja
    legs = [j for j, name in enumerate(joint_names) if any(p in name for p in ("hip", "knee", "ankle"))]
    poses[:, legs] *= args.leg_amplitude
    times = times * args.slowdown

    keyframes, duration = to_keyframes(times, poses, joint_names, args.rate, blend=0.5)

    speeds = np.abs(np.diff(poses, axis=0)) / np.diff(times)[:, None]
    print(f"velocidad articular máxima {speeds.max():.1f} rad/s ({joint_names[speeds.max(axis=0).argmax()]})")

    with open(MOTIONS_PATH, "r", encoding="utf-8") as f:
        motions = json.load(f)

    motions[args.name] = {
        "phase": args.phase,
        "duration": duration,
        "source": args.source,
        "extraction": {"video": Path(args.video).name, "start": args.start, "end": args.end,
                       "slowdown": args.slowdown, "leg_amplitude": args.leg_amplitude},
        "keyframes": keyframes,
    }

    # escritura atómica: una interrupción no deja motions.json a medias
    temporary = MOTIONS_PATH.with_suffix(".json.tmp")

    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(motions, f, ensure_ascii=False, indent=2)

    os.replace(temporary, MOTIONS_PATH)

    print(f"'{args.name}' guardado en {MOTIONS_PATH} ({len(keyframes)} keyframes, {duration} s)")


if __name__ == "__main__":
    main()
