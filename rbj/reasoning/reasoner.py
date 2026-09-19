from typing import Any, Dict, Optional

from .context import ReasoningContext
from .decision import ReasoningDecision
from .rules import ReasoningRules


class Reasoner:

    def __init__(self):
        self.last_decision: Optional[ReasoningDecision] = None

    def reason(
        self,
        world: Dict[str, Any],
        command: Optional[Dict[str, Any]],
        memory: Optional[Dict[str, Any]] = None,
    ) -> ReasoningDecision:

        context = ReasoningContext(
            world=world,
            command=command,
            memory=memory,
        )

        decision = ReasoningRules.evaluate(context)

        self.last_decision = decision

        return decision

    def get_last_decision(self):
        return self.last_decision

    def reset(self):
        self.last_decision = None