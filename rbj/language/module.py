from pathlib import Path

from rbj.language.inference_structured import SJGLanguageStructuredInference
from rbj.language.structured_decision import StructuredLanguageDecision


class LanguageModule:
    """
    Módulo de alto nivel para procesamiento de lenguaje natural.

    Entrada:
        texto natural

    Salida:
        comando estructurado o solicitud de aclaración.

    Implementa el flujo completo de mejoras.txt:
        1. Predicción semántica con el modelo estructurado.
        2. Promedio de confianzas / N clases.
        3. Umbral de 0.89 para aceptación.
        4. Hasta 3 reintentos de predicción si no se supera el umbral.
    """

    def __init__(
        self,
        checkpoint_path,
        threshold: float = 0.89,
        max_retries: int = 3,
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
            threshold=threshold,
            max_retries=max_retries,
        )

    def process(self, text: str) -> dict:
        """
        Procesa una instrucción completa con reintentos automáticos.

        Ejecuta la predicción hasta `max_retries` veces hasta que
        la confianza promedio supere el umbral configurado.

        Returns
        -------
        dict con claves:
            text        : str
            prediction  : dict   (última predicción)
            accepted    : bool
            status      : str
            reason      : str
            command     : dict | None
            confidence  : float
            attempts    : int
        """

        if not isinstance(text, str):
            raise TypeError("text debe ser un string.")

        text = text.strip()

        if not text:
            return {
                "text": text,
                "prediction": None,
                "accepted": False,
                "status": "CLARIFICATION_REQUIRED",
                "reason": "EMPTY_INPUT",
                "command": None,
                "confidence": 0.0,
                "attempts": 0,
            }

        result = self.decision.evaluate_with_retries(
            text=text,
            inference=self.inference,
        )

        return {
            "text": text,
            **result,
        }

    def predict(self, text: str) -> dict:
        """
        Devuelve únicamente la predicción semántica (sin decisión).
        """

        return self.inference.predict(text)

    def decide(self, prediction: dict) -> dict:
        """
        Evalúa una predicción previamente generada (sin reintentos).
        """

        return self.decision.evaluate(prediction)