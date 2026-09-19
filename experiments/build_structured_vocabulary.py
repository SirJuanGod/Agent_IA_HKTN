import json
from pathlib import Path
from sys import path

path.append(str(Path(__file__).parent / ".." / "rbj"))

from language.vocabulary import Vocabulary
from language.tokenizer import Tokenizer


def load_dataset(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def main():

    path = Path(
        "dataset/language/train.json"
    )

    data = load_dataset(path)

    vocab = Vocabulary()

    tokenizer = Tokenizer(vocab)

    for item in data:

        tokens = tokenizer.tokenize(
            item["text"]
        )

        for token in tokens:

            vocab.add_word(token)

    output = Path(
        "checkpoints/"
        "sjg_structured_vocabulary.json"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    vocab.save(output)

    print("=" * 60)
    print("SJG Structured Vocabulary")
    print("=" * 60)

    print(
        f"Training sentences: {len(data)}"
    )

    print(
        f"Vocabulary size: {len(vocab)}"
    )

    print(
        f"Saved: {output}"
    )


if __name__ == "__main__":
    main()