import sys
import os
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from rbj.language.vocabulary import Vocabulary
from rbj.language.tokenizer import Tokenizer


def main():

    vocabulary = Vocabulary()

    sentences = [

        "ve hacia la meta",
        "busca el objetivo",
        "llega al objetivo",

        "muévete arriba",
        "sube",
        "ve arriba",

        "muévete abajo",
        "baja",
        "ve abajo",

        "ve a la izquierda",
        "muévete izquierda",

        "ve a la derecha",
        "muévete derecha",

        "detente",
        "para"
    ]

    # Construir vocabulario

    for sentence in sentences:

        words = sentence.lower().split()

        for word in words:

            vocabulary.add_word(word)

    tokenizer = Tokenizer(
        vocabulary
    )

    print("\n")
    print("=" * 50)
    print("SJG-AGENT 0.3.1")
    print("VOCABULARY + TOKENIZER")
    print("=" * 50)

    print(
        "\nTamaño del vocabulario:",
        len(vocabulary)
    )

    print("\nVOCABULARIO:")

    for word, index in vocabulary.word_to_id.items():

        print(
            f"{index:3} -> {word}"
        )

    sentence = "ve hacia la meta"

    tokens = tokenizer.tokenize(
        sentence
    )

    encoded = tokenizer.encode(
        sentence
    )

    decoded = tokenizer.decode(
        encoded
    )

    print("\nPRUEBA")

    print(
        "Texto:",
        sentence
    )

    print(
        "Tokens:",
        tokens
    )

    print(
        "IDs:",
        encoded
    )

    print(
        "Decodificado:",
        decoded
    )


    unknown_sentence = "camina hacia marte"

    unknown_tokens = tokenizer.tokenize(
        unknown_sentence
    )

    unknown_ids = tokenizer.encode(
        unknown_sentence
    )

    unknown_decoded = tokenizer.decode(
        unknown_ids
    )

    print("\nPALABRAS DESCONOCIDAS")

    print(
        "Texto:",
        unknown_sentence
    )

    print(
        "Tokens:",
        unknown_tokens
    )

    print(
        "IDs:",
        unknown_ids
    )

    print(
        "Decodificado:",
        unknown_decoded
    )

    vocabulary.save(
        "checkpoints/sjg_vocabulary_031.json"
    )

    print(
        "\nVocabulario guardado."
    )

if __name__ == "__main__":

    main()
