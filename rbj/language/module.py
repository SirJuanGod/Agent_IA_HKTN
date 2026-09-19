from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent / ".."))

from rbj.language.inference_structured import SJGLanguageStructuredInference
from rbj.language.structured_decision import StructuredLanguageDecision


class LanguageModule:
    """
    Módulo de alto nivel para procesamiento de lenguaje.

    Entrada:
        texto natural

    Salida:
        comando estructurado o solicitud de aclaración.
    """

    def __init__(
        self,
        checkpoint_path,
        threshold=0.70,
        device=None,
    ):
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"No se encontró el checkpoint: {checkpoint_path}"
            )

        self.inference = SJGLanguageStructuredInference(
            checkpoint_path=str(checkpoint_path),
            device=device,
        )

        self.decision = StructuredLanguageDecision(
            threshold=threshold
        )

    def process(self, text):
        """
        Procesa una instrucción completa.

        Returns:
            dict
        """

        if not isinstance(text, str):
            raise TypeError("text debe ser un string.")

        text = text.strip()

        if not text:
            return {
                "status": "CLARIFICATION_REQUIRED",
                "command": None,
                "reason": "EMPTY_INPUT",
            }

        prediction = self.inference.predict(text)

        decision = self.decision.evaluate(prediction)

        return {
            "text": text,
            "prediction": prediction,
            **decision,
        }

    def predict(self, text):
        """
        Devuelve únicamente la predicción semántica.
        """

        return self.inference.predict(text)

    def decide(self, prediction):
        """
        Evalúa una predicción previamente generada.
        """

        return self.decision.evaluate(prediction)