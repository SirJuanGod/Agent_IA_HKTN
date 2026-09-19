import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / ".."))

from rbj.language.inference_structured import (
    SJGLanguageStructuredInference,
)

from rbj.language.structured_decision import (
    StructuredLanguageDecision,
)


CHECKPOINT = (
    "checkpoints/"
    "sjg_language_037.pt"
)


def print_prediction(
    prediction
):

    print("\nPREDICCIÓN")
    print("-" * 50)

    print(
        f'Texto: {prediction["text"]}'
    )

    for field in [
        "intent",
        "action",
        "direction",
        "target",
        "subject",
        "distance",
    ]:

        confidence = prediction[
            f"{field}_confidence"
        ]

        print(
            f"{field:12}: "
            f"{str(prediction[field]):12} "
            f"({confidence:.3f})"
        )


def main():

    model = (
        SJGLanguageStructuredInference(
            CHECKPOINT
        )
    )

    decision = (
        StructuredLanguageDecision(
            threshold=0.80
        )
    )

    tests = [

        "ve hacia la derecha",

        "quiero que el robot avance "
        "tres casillas hacia la izquierda",

        "haz que el robot suba dos pasos",

        "lleva el robot hasta la meta",

        "dirígete hacia el objetivo",

        "detente",

        "avanza",

        "muévete un poco",

        "ve por ese lado",
    ]

    print("=" * 70)
    print("SJG-Agent - Structured Language Test")
    print("=" * 70)

    for text in tests:

        prediction = model.predict(
            text
        )

        print_prediction(
            prediction
        )

        result = decision.evaluate(
            prediction
        )

        print("\nDECISIÓN")
        print("-" * 50)

        print(
            f"Status: "
            f"{result['status']}"
        )

        print(
            f"Accepted: "
            f"{result['accepted']}"
        )

        print(
            f"Confidence: "
            f"{result['confidence']:.3f}"
        )

        print(
            f"Reason: "
            f"{result['reason']}"
        )

        print(
            f"Command: "
            f"{result['command']}"
        )


if __name__ == "__main__":
    main()