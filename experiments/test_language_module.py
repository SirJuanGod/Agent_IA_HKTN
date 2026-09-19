import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / ".."))

from rbj.language import LanguageModule


CHECKPOINT = "checkpoints/sjg_language_037.pt"


def print_result(result):
    print("\n" + "=" * 70)

    print("TEXTO:")
    print(result["text"])

    print("\nESTADO:")
    print(result["status"])

    print("\nRAZÓN:")
    print(result.get("reason"))

    print("\nCOMANDO:")
    print(result.get("command"))

    if "prediction" in result:
        prediction = result["prediction"]

        print("\nPREDICCIÓN SEMÁNTICA:")

        for field in [
            "intent",
            "action",
            "direction",
            "target",
            "subject",
            "distance",
        ]:
            value = prediction.get(field)
            confidence = prediction.get(
                f"{field}_confidence"
            )

            print(
                f"  {field:10s}: "
                f"{str(value):10s} "
                f"(confianza={confidence:.4f})"
            )


def main():

    language = LanguageModule(
        checkpoint_path=CHECKPOINT,
        threshold=0.70,
    )

    commands = [
        "ve hacia la derecha",
        "quiero que el robot avance tres casillas hacia la izquierda",
        "haz que el robot suba dos pasos",
        "lleva el robot hasta la meta",
        "dirígete hacia el objetivo",
        "detente",
        "avanza",
        "muévete un poco",
        "ve por ese lado",
    ]

    for text in commands:

        result = language.process(text)

        print_result(result)


if __name__ == "__main__":
    main()