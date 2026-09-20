"""
Módulo de detección de ambigüedad lingüística.

Implementa el requisito de mejoras.txt:
    "Mejorar la percepción de la ambigüedad — que el agente pueda
    predecir si una sentencia es ambigua o no."

Una sentencia se considera AMBIGUA cuando:
    - La intención no está clara (podría ser MOVE o GO_TO).
    - Faltan parámetros esenciales (dirección, objetivo).
    - Contiene lenguaje contradictorio o incompleto.
    - El modelo semántico devuelve confianza baja (< threshold).

Una sentencia es CLARA cuando el modelo puede extraer todos los
campos con confianza suficiente.
"""

from dataclasses import dataclass
from typing import Optional


# ============================================================
# Resultado de detección
# ============================================================

@dataclass
class AmbiguityResult:
    """
    Resultado del análisis de ambigüedad.

    Atributos
    ----------
    is_ambiguous : bool
        True si la sentencia es ambigua.
    reason : str
        Código explicando por qué es ambigua (o "CLEAR").
    confidence : float
        Confianza promedio del modelo semántico.
    intent : str | None
        Intención detectada (aunque sea con baja confianza).
    suggestion : str | None
        Sugerencia de aclaración para mostrar al usuario.
    """

    is_ambiguous: bool
    reason: str
    confidence: float
    intent: Optional[str] = None
    suggestion: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "is_ambiguous": self.is_ambiguous,
            "reason": self.reason,
            "confidence": self.confidence,
            "intent": self.intent,
            "suggestion": self.suggestion,
        }


# ============================================================
# Razones de ambigüedad
# ============================================================

class AmbiguityReason:
    CLEAR                   = "CLEAR"
    EMPTY_INPUT             = "EMPTY_INPUT"
    UNKNOWN_INTENT          = "UNKNOWN_INTENT"
    MISSING_DIRECTION       = "MISSING_DIRECTION"
    MISSING_TARGET          = "MISSING_TARGET"
    LOW_OVERALL_CONFIDENCE  = "LOW_OVERALL_CONFIDENCE"
    CONFLICTING_FIELDS      = "CONFLICTING_FIELDS"


# ============================================================
# Sugerencias de aclaración
# ============================================================

_SUGGESTIONS = {
    AmbiguityReason.UNKNOWN_INTENT: (
        "No entendí qué quieres hacer. "
        "¿Quieres que me mueva en una dirección o que vaya a la meta?"
    ),
    AmbiguityReason.MISSING_DIRECTION: (
        "¿En qué dirección quieres que me mueva? "
        "(arriba, abajo, izquierda, derecha)"
    ),
    AmbiguityReason.MISSING_TARGET: (
        "¿A dónde quieres que vaya? ¿A la meta?"
    ),
    AmbiguityReason.LOW_OVERALL_CONFIDENCE: (
        "No estoy seguro de entender el comando. "
        "¿Puedes repetirlo de otra manera?"
    ),
    AmbiguityReason.CONFLICTING_FIELDS: (
        "El comando contiene información contradictoria. "
        "¿Puedes ser más específico?"
    ),
}


# ============================================================
# Detector principal
# ============================================================

class AmbiguityDetector:
    """
    Detecta si una sentencia de lenguaje natural es ambigua.

    Usa la salida del motor de inferencia estructurado para
    evaluar la confianza en cada campo semántico.

    Parámetros
    ----------
    threshold : float
        Confianza mínima por campo para considerar la sentencia
        como CLARA. Defecto: 0.70 (más permisivo que el umbral
        de decisión de 0.89, ya que la ambigüedad es un diagnóstico
        más temprano).
    """

    def __init__(self, threshold: float = 0.70):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold debe estar entre 0.0 y 1.0.")

        self.threshold = threshold

    # ----------------------------------------------------------
    # API principal
    # ----------------------------------------------------------

    def is_ambiguous(self, text: str, inference) -> bool:
        """
        Devuelve True si la sentencia es ambigua.

        Parámetros
        ----------
        text : str
            Texto de entrada.
        inference : SJGLanguageStructuredInference
            Motor de inferencia semántica.
        """

        result = self.analyze(text, inference)
        return result.is_ambiguous

    def analyze(self, text: str, inference) -> AmbiguityResult:
        """
        Analiza la sentencia y devuelve un resultado detallado.

        Parámetros
        ----------
        text : str
            Texto de entrada.
        inference : SJGLanguageStructuredInference
            Motor de inferencia semántica.

        Returns
        -------
        AmbiguityResult
        """

        text = text.strip() if text else ""

        if not text:
            return AmbiguityResult(
                is_ambiguous=True,
                reason=AmbiguityReason.EMPTY_INPUT,
                confidence=0.0,
                suggestion="Por favor, introduce un comando.",
            )

        prediction = inference.predict(text)

        return self._evaluate(prediction)

    def analyze_prediction(self, prediction: dict) -> AmbiguityResult:
        """
        Analiza una predicción ya generada (sin re-inferir).

        Útil cuando el módulo de lenguaje ya ha llamado a `predict`.
        """

        return self._evaluate(prediction)

    # ----------------------------------------------------------
    # Evaluación interna
    # ----------------------------------------------------------

    def _evaluate(self, prediction: dict) -> AmbiguityResult:

        intent          = prediction.get("intent")
        intent_conf     = prediction.get("intent_confidence", 0.0)
        direction_conf  = prediction.get("direction_confidence", 0.0)
        target_conf     = prediction.get("target_confidence", 0.0)
        action_conf     = prediction.get("action_confidence", 0.0)

        # Confianza promedio general
        fields = [
            "intent_confidence",
            "action_confidence",
            "direction_confidence",
            "target_confidence",
            "subject_confidence",
            "distance_confidence",
        ]
        avg_confidence = sum(
            prediction.get(f, 0.0) for f in fields
        ) / len(fields)

        # -------------------------------------------------
        # 1. Intent desconocida
        # -------------------------------------------------

        if intent_conf < self.threshold:
            return AmbiguityResult(
                is_ambiguous=True,
                reason=AmbiguityReason.UNKNOWN_INTENT,
                confidence=avg_confidence,
                intent=intent,
                suggestion=_SUGGESTIONS[AmbiguityReason.UNKNOWN_INTENT],
            )

        # -------------------------------------------------
        # 2. MOVE sin dirección clara
        # -------------------------------------------------

        if intent == "MOVE" and direction_conf < self.threshold:
            return AmbiguityResult(
                is_ambiguous=True,
                reason=AmbiguityReason.MISSING_DIRECTION,
                confidence=avg_confidence,
                intent=intent,
                suggestion=_SUGGESTIONS[AmbiguityReason.MISSING_DIRECTION],
            )

        # -------------------------------------------------
        # 3. GO_TO sin objetivo claro
        # -------------------------------------------------

        if intent == "GO_TO" and target_conf < self.threshold:
            return AmbiguityResult(
                is_ambiguous=True,
                reason=AmbiguityReason.MISSING_TARGET,
                confidence=avg_confidence,
                intent=intent,
                suggestion=_SUGGESTIONS[AmbiguityReason.MISSING_TARGET],
            )

        # -------------------------------------------------
        # 4. MOVE con dirección ≠ NONE pero acción con
        #    confianza baja → campos contradictorios
        # -------------------------------------------------

        if (
            intent == "MOVE"
            and action_conf < self.threshold
            and direction_conf >= self.threshold
        ):
            return AmbiguityResult(
                is_ambiguous=True,
                reason=AmbiguityReason.CONFLICTING_FIELDS,
                confidence=avg_confidence,
                intent=intent,
                suggestion=_SUGGESTIONS[AmbiguityReason.CONFLICTING_FIELDS],
            )

        # -------------------------------------------------
        # 5. Confianza general baja (todos los campos dudosos)
        # -------------------------------------------------

        if avg_confidence < self.threshold:
            return AmbiguityResult(
                is_ambiguous=True,
                reason=AmbiguityReason.LOW_OVERALL_CONFIDENCE,
                confidence=avg_confidence,
                intent=intent,
                suggestion=_SUGGESTIONS[AmbiguityReason.LOW_OVERALL_CONFIDENCE],
            )

        # -------------------------------------------------
        # Sentencia CLARA
        # -------------------------------------------------

        return AmbiguityResult(
            is_ambiguous=False,
            reason=AmbiguityReason.CLEAR,
            confidence=avg_confidence,
            intent=intent,
        )
