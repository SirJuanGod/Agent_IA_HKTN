from pathlib import Path
from sys import path

path.append(str(Path(__file__).parent / ".." / "rbj"))

from language.structured_dataset import (
    StructuredLanguageGenerator,
    save_examples,
    save_json,
)


def main():

    print("=" * 70)
    print("SJG-Agent")
    print("0.3.6 - Structured Language Dataset")
    print("=" * 70)

    generator = StructuredLanguageGenerator(
        seed=42
    )

    print("\nGenerando ejemplos...")

    examples = generator.generate_all()

    print(
        f"Ejemplos generados: {len(examples)}"
    )

    train, validation, test = (
        generator.split_dataset(examples)
    )

    ambiguous = (
        generator.generate_ambiguous()
    )

    output = Path(
        "dataset/language"
    )

    output.mkdir(
        parents=True,
        exist_ok=True
    )

    save_examples(
        train,
        output / "train.json"
    )

    save_examples(
        validation,
        output / "validation.json"
    )

    save_examples(
        test,
        output / "test.json"
    )

    save_json(
        ambiguous,
        output / "test_ambiguous.json"
    )

    print("\nDataset generado:")
    print("-" * 40)

    print(
        f"Train:       {len(train)}"
    )

    print(
        f"Validation:  {len(validation)}"
    )

    print(
        f"Test:        {len(test)}"
    )

    print(
        f"Ambiguous:   {len(ambiguous)}"
    )

    print("\nArchivos:")
    print(output / "train.json")
    print(output / "validation.json")
    print(output / "test.json")
    print(output / "test_ambiguous.json")


if __name__ == "__main__":
    main()