from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PlanStep:
    action: str
    direction: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "direction": self.direction,
            "parameters": self.parameters,
        }


@dataclass
class Plan:
    status: str
    goal: Optional[str] = None
    steps: List[PlanStep] = field(default_factory=list)
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.steps)

    def add_step(self, step: PlanStep):
        self.steps.append(step)

    def next_step(self) -> Optional[PlanStep]:
        if not self.steps:
            return None

        return self.steps[0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "goal": self.goal,
            "steps": [
                step.to_dict()
                for step in self.steps
            ],
            "reason": self.reason,
            "metadata": self.metadata,
        }