import json
from collections import Counter
from pathlib import Path


def load(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def distribution(data, field):

    return Counter(
        item[field]
        for item in data
    )


def print_distribution(
    title,
    counter
):

    print(f"\n{title}")
    print("-" * 45)

    for key, value in sorted(
        counter.items(),
        key=lambda x: str(x[0])
    ):

        print(
            f"{str(key):25} {value}"
        )


def inspect_dataset(
    name,
    data
):

    print("\n")
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Total: {len(data)}"
    )

    print_distribution(
        "INTENTS",
        distribution(
            data,
            "intent"
        )
    )

    print_distribution(
        "DIRECTIONS",
        distribution(
            data,
            "direction"
        )
    )

    print_distribution(
        "DISTANCES",
        distribution(
            data,
            "distance"
        )
    )

    print_distribution(
        "DIFFICULTY",
        distribution(
            data,
            "difficulty"
        )
    )

    print_distribution(
        "TEMPLATE FAMILIES",
        distribution(
            data,
            "template_family"
        )
    )


def show_samples(
    data,
    amount=15
):

    print("\nMUESTRAS")
    print("-" * 70)

    for item in data[:amount]:

        print()
        print(
            f'Texto:      {item["text"]}'
        )

        print(
            f'Intent:     {item["intent"]}'
        )

        print(
            f'Direction:  {item["direction"]}'
        )

        print(
            f'Target:     {item["target"]}'
        )

        print(
            f'Distance:   {item["distance"]}'
        )

        print(
            f'Difficulty: {item["difficulty"]}'
        )

        print(
            f'Family:     {item["template_family"]}'
        )


def check_duplicates(data):

    texts = [
        item["text"].strip().lower()
        for item in data
    ]

    duplicates = [
        text
        for text, count
        in Counter(texts).items()
        if count > 1
    ]

    print("\nDUPLICADOS")
    print("-" * 45)

    if not duplicates:

        print("No hay duplicados.")

    else:

        print(
            f"Duplicados encontrados: "
            f"{len(duplicates)}"
        )

        for text in duplicates[:20]:

            print(
                f"- {text}"
            )


def main():

    base = Path(
        "dataset/language"
    )

    train = load(
        base / "train.json"
    )

    validation = load(
        base / "validation.json"
    )

    test = load(
        base / "test.json"
    )

    ambiguous = load(
        base / "test_ambiguous.json"
    )

    inspect_dataset(
        "TRAIN",
        train
    )

    show_samples(
        train
    )

    inspect_dataset(
        "VALIDATION",
        validation
    )

    show_samples(
        validation
    )

    inspect_dataset(
        "TEST",
        test
    )

    show_samples(
        test
    )

    print("\n")
    print("=" * 70)
    print("AMBIGUOUS TEST")
    print("=" * 70)

    for item in ambiguous:

        print(
            f'{item["text"]:<35}'
            f' → {item["reason"]}'
        )

    check_duplicates(train)
    check_duplicates(validation)
    check_duplicates(test)


if __name__ == "__main__":
    main()