class LanguageDecision:

    def __init__(self, threshold=0.70):

        self.threshold = threshold

    def decide(self, result):

        confidence = result["confidence"]

        if confidence < self.threshold:

            return {
                "accepted": False,
                "intent": "UNKNOWN",
                "confidence": confidence,
            }

        return {
            "accepted": True,
            "intent": result["intent"],
            "confidence": confidence,
        }