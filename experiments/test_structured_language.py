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
    "sjg_language_038.pt"
)

# texto -> comando esperado (incluye frases que no están en train)
EXPECTED = {
    "saluda con la mano": {"action": "PERFORM", "motion": "wave"},
    "por favor levanta los brazos": {"action": "PERFORM", "motion": "raise_arms"},
    "necesito que el robot señale a la izquierda": {"action": "PERFORM", "motion": "point_left"},
    "me gustaría que mi robot gire la cintura": {"action": "PERFORM", "motion": "twist_waist"},
    "agáchate": {"action": "PERFORM", "motion": "squat"},
    "haz una reverencia": {"action": "PERFORM", "motion": "bow"},
    "ve hacia la derecha": {"action": "MOVE", "direction": "RIGHT", "distance": 1},
    "lleva el robot hasta la meta": {"action": "GO_TO", "target": "GOAL"},
    "detente": {"action": "STOP"},
}


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

    for text, command in EXPECTED.items():

        result = decision.evaluate(
            model.predict(text)
        )

        assert result["command"] == command, (
            text, result
        )

    print("\ntest_structured_language OK")


if __name__ == "__main__":
    main()