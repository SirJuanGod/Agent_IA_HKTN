import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / ".."))

from rbj.language.inference_structured import (
    SJGLanguageStructuredInference,
)


CHECKPOINT = (
    "checkpoints/"
    "sjg_language_037.pt"
)


FIELDS = [
    "intent",
    "action",
    "direction",
    "target",
    "subject",
    "distance",
]


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

    data = load_json(
        "dataset/language/test.json"
    )

    correct = {
        field: 0
        for field in FIELDS
    }

    exact = 0

    total = len(data)

    print("=" * 70)
    print("SJG-Agent - Structured Language Evaluation")
    print("=" * 70)

    for item in data:

        prediction = model.predict(
            item["text"]
        )

        sample_exact = True

        for field in FIELDS:

            expected = item[field]
            predicted = prediction[field]

            if expected == predicted:

                correct[field] += 1

            else:

                sample_exact = False

        if sample_exact:

            exact += 1

    print(
        f"\nTotal samples: {total}"
    )

    print("\nFIELD ACCURACY")
    print("-" * 50)

    for field in FIELDS:

        accuracy = (
            correct[field]
            / total
        )

        print(
            f"{field:12}: "
            f"{accuracy:.4f}"
        )

    exact_accuracy = (
        exact / total
    )

    print("\nEXACT SEMANTIC MATCH")
    print("-" * 50)

    print(
        f"{exact_accuracy:.4f}"
    )

    print(
        f"{exact}/{total}"
    )


if __name__ == "__main__":
    main()