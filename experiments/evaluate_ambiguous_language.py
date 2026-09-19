import json
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


def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def main():

    model = (
        SJGLanguageStructuredInference(
            CHECKPOINT
        )
    )

    decision = (
        StructuredLanguageDecision(
            threshold=0.70
        )
    )

    data = load_json(
        "dataset/language/"
        "test_ambiguous.json"
    )

    correct = 0

    print("=" * 70)
    print("SJG-Agent - Ambiguous Language Evaluation")
    print("=" * 70)

    for item in data:

        prediction = model.predict(
            item["text"]
        )

        result = decision.evaluate(
            prediction
        )

        is_correct = not result[
            "accepted"
        ]

        if is_correct:

            correct += 1

        print("\n")
        print(
            f'Text: {item["text"]}'
        )

        print(
            f'Expected: {item["expected"]}'
        )

        print(
            f'Predicted intent: '
            f'{prediction["intent"]}'
        )

        print(
            f'Predicted direction: '
            f'{prediction["direction"]}'
        )

        print(
            f'Accepted: '
            f'{result["accepted"]}'
        )

        print(
            f'Reason: '
            f'{result["reason"]}'
        )

    accuracy = (
        correct / len(data)
    )

    print("\n")
    print("=" * 70)

    print(
        f"Ambiguity rejection accuracy: "
        f"{accuracy:.4f}"
    )


if __name__ == "__main__":
    main()