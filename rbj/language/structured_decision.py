"""
Módulo de decisión para el modelo de lenguaje estructurado.

Implementa la lógica de aceptación/rechazo de predicciones
descrita en mejoras.txt:

    - El promedio de confianzas se divide entre el número de
      campos evaluados (N clases de predicción).
    - El valor debe superar el umbral de 0.89 para ser aceptado.
    - Si no supera el umbral, se reintenta hasta 3 veces usando
      el método `evaluate_with_retries`, que re-lanza la predicción
      en el motor de inferencia antes de decidir.
"""


class StructuredLanguageDecision:
    """
    Evalúa si una predicción semántica es suficientemente
    confiable para convertirse en un comando aceptado.

    Parámetros
    ----------
    threshold : float
        Umbral mínimo de confianza (defecto 0.89, según mejoras.txt).
    max_retries : int
        Número máximo de reintentos si la predicción no supera el umbral.
    """

    def __init__(
        self,
        threshold: float = 0.89,
        max_retries: int = 3,
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold debe estar entre 0.0 y 1.0.")

        self.threshold = threshold
        self.max_retries = max_retries

    # =========================================================
    # Confianza promediada por número de campos (N clases)
    # =========================================================

    def _average_confidence(self, prediction) -> float:
        """
        Calcula la confianza promedio dividiendo la suma de
        confianzas entre el número de campos evaluados (N clases),
        tal como especifica mejoras.txt.
        """

        fields = [
            "intent_confidence",
            "action_confidence",
            "direction_confidence",
            "target_confidence",
            "subject_confidence",
            "distance_confidence",
        ]

        confidences = [prediction[f] for f in fields]

        n_classes = len(confidences)

        return sum(confidences) / n_classes

    # =========================================================
    # Evaluación de una sola predicción
    # =========================================================

    def evaluate(self, prediction) -> dict:
        """
        Evalúa una predicción y devuelve si fue aceptada.

        Returns
        -------
        dict con claves:
            accepted  : bool
            status    : str
            reason    : str
            command   : dict | None
            confidence: float
        """

        avg_confidence = self._average_confidence(prediction)

        intent = prediction["intent"]

        # -------------------------------------------------
        # STOP — solo requiere confianza en intent
        # -------------------------------------------------

        if intent == "STOP":

            if (
                prediction["intent_confidence"]
                >= self.threshold
            ):

                return {
                    "accepted": True,
                    "status": "READY",
                    "reason": "STOP_COMMAND",
                    "command": {
                        "action": "STOP"
                    },
                    "confidence": avg_confidence,
                }

        # -------------------------------------------------
        # GO_TO — requiere intent + action + target
        # -------------------------------------------------

        if intent == "GO_TO":

            required = [
                prediction["intent_confidence"],
                prediction["action_confidence"],
                prediction["target_confidence"],
            ]

            required_avg = sum(required) / len(required)

            if required_avg >= self.threshold:

                return {
                    "accepted": True,
                    "status": "READY",
                    "reason": "GO_TO_COMMAND",
                    "command": {
                        "action": "GO_TO",
                        "target": prediction["target"],
                    },
                    "confidence": required_avg,
                }

        # -------------------------------------------------
        # MOVE — requiere intent + action + direction + distance
        # -------------------------------------------------

        if intent == "MOVE":

            required = [
                prediction["intent_confidence"],
                prediction["action_confidence"],
                prediction["direction_confidence"],
                prediction["distance_confidence"],
            ]

            required_avg = sum(required) / len(required)

            if required_avg >= self.threshold:

                return {
                    "accepted": True,
                    "status": "READY",
                    "reason": "MOVE_COMMAND",
                    "command": {
                        "action": "MOVE",
                        "direction": prediction["direction"],
                        "distance": prediction["distance"],
                    },
                    "confidence": required_avg,
                }

        # -------------------------------------------------
        # No superó el umbral
        # -------------------------------------------------

        return {
            "accepted": False,
            "status": "CLARIFICATION_REQUIRED",
            "reason": self._clarification_reason(prediction),
            "command": None,
            "confidence": avg_confidence,
        }

    # =========================================================
    # Evaluación con reintentos (hasta max_retries = 3)
    # =========================================================

    def evaluate_with_retries(
        self,
        text: str,
        inference,
    ) -> dict:
        """
        Intenta la predicción hasta `max_retries` veces.

        Tal como especifica mejoras.txt:
            "debe volver a pasar y hacer la prediccion 3 veces
            hasta que sea aceptable".

        Parámetros
        ----------
        text : str
            Texto de entrada.
        inference : SJGLanguageStructuredInference
            Motor de inferencia con método `predict(text)`.

        Returns
        -------
        dict con el mismo esquema que `evaluate()`, más:
            attempts : int  → número de intentos realizados
        """

        last_result = None

        for attempt in range(1, self.max_retries + 1):

            prediction = inference.predict(text)

            result = self.evaluate(prediction)

            result["attempts"] = attempt
            result["prediction"] = prediction

            if result["accepted"]:
                return result

            last_result = result

        # Devolver el último resultado aunque no fuera aceptado
        return last_result

    # =========================================================
    # Razón de clarificación
    # =========================================================

    def _clarification_reason(self, prediction) -> str:

        if prediction["intent_confidence"] < self.threshold:
            return "UNKNOWN_INTENT"

        if prediction["intent"] == "MOVE":
            if prediction["direction_confidence"] < self.threshold:
                return "MISSING_OR_UNCERTAIN_DIRECTION"

        if prediction["intent"] == "GO_TO":
            if prediction["target_confidence"] < self.threshold:
                return "UNCERTAIN_TARGET"

        return "LOW_CONFIDENCE"