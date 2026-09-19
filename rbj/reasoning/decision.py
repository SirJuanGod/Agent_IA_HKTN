from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ReasoningDecision:
    decision: str
    reason: str

    confidence: float = 1.0

    action: Optional[str] = None
    direction: Optional[str] = None
    target: Optional[str] = None

    alternative: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "confidence": self.confidence,
            "action": self.action,
            "direction": self.direction,
            "target": self.target,
            "alternative": self.alternative,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"ReasoningDecision("
            f"decision={self.decision!r}, "
            f"reason={self.reason!r}, "
            f"confidence={self.confidence:.2f})"
        )