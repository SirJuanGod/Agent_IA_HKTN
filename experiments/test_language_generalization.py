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
from rbj.language.decision import LanguageDecision


CHECKPOINT = (
    "checkpoints/sjg_language_036.pt"
)


def main():

    language = SJGLanguageInference(
        CHECKPOINT
    )

    decision = LanguageDecision(threshold=0.80)

    test_cases = [

        # GO_TO_GOAL
        "quiero que el robot llegue al objetivo",
        "lleva la máquina hasta la meta",
        "haz que el robot vaya hacia el objetivo",

        # MOVE_UP
        "quiero que subas",
        "desplázate en dirección norte",
        "haz que el robot avance hacia arriba",

        # MOVE_DOWN
        "quiero que bajes",
        "mueve el robot hacia abajo",
        "avanza en dirección inferior",

        # MOVE_LEFT
        "quiero que vayas a la izquierda",
        "mueve la máquina hacia la izquierda",
        "desplázate hacia el lado izquierdo",

        # MOVE_RIGHT
        "quiero que vayas a la derecha",
        "mueve la máquina hacia la derecha",
        "desplázate hacia el lado derecho",

        # STOP
        "quiero que te detengas",
        "deja de moverte",
        "quiero que el robot permanezca quieto",
    ]

    print("=" * 70)
    print("SJG-AGENT - GENERALIZATION TEST")
    print("=" * 70)

    for sentence in test_cases:

        result = language.predict(
            sentence
        )
        
        final_result = decision.decide(result)

        print()
        print(
            f"Texto: {sentence}"
        )

        print(
            f"Intent: {final_result['intent']} "
            f"({final_result['accepted']})"
        )

        print(
            f"Confidence: "
            f"{final_result['confidence']:.3f}"
        )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()