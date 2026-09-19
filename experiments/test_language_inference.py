import os
import sys
import warnings

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from rbj.language.inference import SJGLanguageInference


CHECKPOINT = "checkpoints/sjg_language_034.pt"


def main():

    print("=" * 60)
    print("SJG-AGENT - LANGUAGE INFERENCE 0.3.5")
    print("=" * 60)

    # --------------------------------------------------
    # Cargar modelo
    # --------------------------------------------------

    language = SJGLanguageInference(
        CHECKPOINT
    )

    print()
    print("Modelo cargado correctamente.")

    print(
        "Device:",
        language.device
    )

    # --------------------------------------------------
    # Frases de prueba
    # --------------------------------------------------

    test_sentences = [

        "ve hacia la meta",

        "sube",

        "quiero que subas",

        "ve hacia abajo",

        "muévete a la izquierda",

        "desplázate hacia la derecha",

        "para",

        "quédate quieto",

        "dirígete al objetivo",

        "camina hasta el objetivo",
    ]

    # --------------------------------------------------
    # Predicciones
    # --------------------------------------------------

    print()
    print("-" * 60)
    print("PREDICTIONS")
    print("-" * 60)

    for sentence in test_sentences:

        result = language.predict(
            sentence
        )

        print()
        print("Texto:")
        print(
            f"  {result['text']}"
        )

        print(
            f"Intent: "
            f"{result['intent']}"
        )

        print(
            f"Confidence: "
            f"{result['confidence']:.4f}"
        )

    # --------------------------------------------------
    # Top-K
    # --------------------------------------------------

    print()
    print("-" * 60)
    print("TOP-K")
    print("-" * 60)

    sentence = (
        "quiero que el robot vaya "
        "hacia la meta"
    )

    print()
    print("Texto:")
    print(sentence)

    predictions = language.predict_top_k(
        sentence,
        k=3,
    )

    for i, prediction in enumerate(
        predictions,
        start=1,
    ):

        print(
            f"{i}. "
            f"{prediction['intent']} "
            f"→ "
            f"{prediction['confidence']:.4f}"
        )

    print()
    print("=" * 60)
    print("INFERENCE OK")
    print("=" * 60)


if __name__ == "__main__":
    main()