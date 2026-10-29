import json

import numpy as np


class MotionLibrary:
    """
    Movimientos definidos por keyframes (config/motions.json).

    Cada keyframe es una pose completa: las articulaciones no listadas
    valen 0 (pose por defecto). Los movimientos son periódicos: se cierra
    el ciclo volviendo al primer keyframe en t = duration.
    """

    def __init__(self, path, joint_names, joint_ranges=None):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.joint_names = list(joint_names)
        index = {name: i for i, name in enumerate(self.joint_names)}

        self.names = list(data.keys())
        self.phases = []
        self.durations = []
        self._times = []
        self._poses = []

        for name in self.names:
            motion = data[name]
            duration = float(motion["duration"])
            keyframes = motion["keyframes"]

            times = [float(k["t"]) for k in keyframes] + [duration]

            if times[0] != 0.0 or np.any(np.diff(times) <= 0):
                raise ValueError(
                    f"Movimiento '{name}': los keyframes deben empezar en t=0, "
                    f"ser crecientes y terminar antes de duration={duration}"
                )

            poses = np.zeros((len(keyframes) + 1, len(self.joint_names)))

            for k, keyframe in enumerate(keyframes):
                for joint, value in keyframe["joints"].items():
                    if joint not in index:
                        raise ValueError(f"Movimiento '{name}': articulación desconocida '{joint}'")

                    if joint_ranges is not None:
                        low, high = joint_ranges[index[joint]]

                        if not low <= value <= high:
                            raise ValueError(
                                f"Movimiento '{name}': {joint}={value} fuera de rango [{low}, {high}]"
                            )

                    poses[k, index[joint]] = value

            poses[-1] = poses[0]

            self.phases.append(motion.get("phase", "fixed"))
            self.durations.append(duration)
            self._times.append(np.array(times))
            self._poses.append(poses)

    def index(self, name):
        return self.names.index(name)

    def ref(self, motion_idx, t):
        """
        Devuelve (q_ref, qd_ref) del movimiento en el instante t.
        """

        times = self._times[motion_idx]
        poses = self._poses[motion_idx]

        # shortcut: interpolación lineal, pasar a cúbica si los movimientos se ven bruscos
        t = t % times[-1]
        k = min(np.searchsorted(times, t, side="right") - 1, len(times) - 2)
        slope = (poses[k + 1] - poses[k]) / (times[k + 1] - times[k])

        return poses[k] + slope * (t - times[k]), slope
