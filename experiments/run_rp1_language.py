import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from rbj.environment.rp1_mujoco import RP1Environment
from rbj.language.module import LanguageModule


ROOT = Path(__file__).resolve().parent.parent
LANGUAGE_CHECKPOINT = ROOT / "checkpoints" / "sjg_language_038.pt"


def main():
    parser = argparse.ArgumentParser(description="Lenguaje natural → movimientos de RP1")
    parser.add_argument("--phase", choices=["fixed", "floating"], default="fixed")
    args = parser.parse_args()

    policy = ROOT / "checkpoints" / f"rp1_motions_{args.phase}.zip"

    language = LanguageModule(str(LANGUAGE_CHECKPOINT))
    robot = RP1Environment(
        policy,
        str(policy).replace(".zip", "_vecnorm.pkl"),
        fixed_base=args.phase == "fixed",
        viewer=True,
    )
    robot.reset()
    trained = [robot.env.library.names[i] for i in robot.env.motions]

    print("Escribe una orden (p. ej. 'saluda con la mano'). Vacío para salir.")

    while True:
        text = input("> ").strip()

        if not text:
            break

        result = language.process(text)

        if result["status"] != "READY":
            print(f"No entendí ({result['reason']}, confianza {result['confidence']:.2f})")
            continue

        command = result["command"]
        print(f"Comando: {command}")

        if command["action"] not in ("PERFORM", "STOP"):
            print("RP1 solo ejecuta movimientos (PERFORM) o STOP")
            continue

        if command["action"] == "PERFORM" and command["motion"] not in trained:
            print(f"'{command['motion']}' no está entrenado en la fase {args.phase}")
            continue

        print(robot.step(command))

        if robot.is_done():
            print("RP1 se cayó, reiniciando")
            robot.reset()

    robot.close()


if __name__ == "__main__":
    main()
