class StructuredLanguageDecision:

    def __init__(
        self,
        threshold=0.80,
    ):

        self.threshold = threshold

    def _confidence_fields(
        self,
        prediction,
    ):

        return [
            prediction["intent_confidence"],
            prediction["action_confidence"],
            prediction["direction_confidence"],
            prediction["target_confidence"],
            prediction["subject_confidence"],
            prediction["distance_confidence"],
        ]

    def evaluate(
        self,
        prediction,
    ):

        confidences = (
            self._confidence_fields(
                prediction
            )
        )

        minimum_confidence = min(
            confidences
        )

        average_confidence = (
            sum(confidences)
            / len(confidences)
        )

        # -------------------------------------------------
        # Comprobar intención
        # -------------------------------------------------

        intent = prediction["intent"]

        # -------------------------------------------------
        # STOP
        # -------------------------------------------------

        if intent == "STOP":

            if (
                prediction[
                    "intent_confidence"
                ] >= self.threshold
            ):

                return {
                    "accepted": True,
                    "status": "READY",
                    "reason": "STOP_COMMAND",
                    "command": {
                        "action": "STOP"
                    },
                    "confidence": average_confidence,
                }

        # -------------------------------------------------
        # GO_TO
        # -------------------------------------------------

        if intent == "GO_TO":

            required = [
                prediction[
                    "intent_confidence"
                ],
                prediction[
                    "action_confidence"
                ],
                prediction[
                    "target_confidence"
                ],
            ]

            if min(required) >= self.threshold:

                return {
                    "accepted": True,
                    "status": "READY",
                    "reason": "GO_TO_COMMAND",
                    "command": {
                        "action": "GO_TO",
                        "target":
                            prediction["target"],
                    },
                    "confidence": (
                        sum(required)
                        / len(required)
                    ),
                }

        # -------------------------------------------------
        # MOVE
        # -------------------------------------------------

        if intent == "MOVE":

            required = [
                prediction[
                    "intent_confidence"
                ],
                prediction[
                    "action_confidence"
                ],
                prediction[
                    "direction_confidence"
                ],
                prediction[
                    "distance_confidence"
                ],
            ]

            if min(required) >= self.threshold:

                return {
                    "accepted": True,
                    "status": "READY",
                    "reason": "MOVE_COMMAND",
                    "command": {
                        "action": "MOVE",
                        "direction":
                            prediction[
                                "direction"
                            ],
                        "distance":
                            prediction[
                                "distance"
                            ],
                    },
                    "confidence": (
                        sum(required)
                        / len(required)
                    ),
                }

        # -------------------------------------------------
        # NO SUFICIENTE INFORMACIÓN
        # -------------------------------------------------

        return {
            "accepted": False,
            "status": "CLARIFICATION_REQUIRED",
            "reason":
                self._clarification_reason(
                    prediction
                ),
            "command": None,
            "confidence":
                minimum_confidence,
        }

    # =====================================================
    # EXPLICACIÓN
    # =====================================================

    def _clarification_reason(
        self,
        prediction,
    ):

        if prediction[
            "intent_confidence"
        ] < self.threshold:

            return "UNKNOWN_INTENT"

        if (
            prediction["intent"]
            == "MOVE"
        ):

            if prediction[
                "direction_confidence"
            ] < self.threshold:

                return "MISSING_OR_UNCERTAIN_DIRECTION"

        if (
            prediction["intent"]
            == "GO_TO"
        ):

            if prediction[
                "target_confidence"
            ] < self.threshold:

                return "UNCERTAIN_TARGET"

        return "LOW_CONFIDENCE"