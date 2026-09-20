from typing import Any, Dict, Optional

from rbj.reasoning import ReasoningDecision

from .grid_planner import GridPathPlanner
from .plan import Plan, PlanStep
from .path_planner import PathPlanner


class Planner:

    def __init__(
        self,
        path_planner: Optional[PathPlanner] = None,
    ):

        self.path_planner = (
            path_planner
            or GridPathPlanner()
        )

        self.last_plan: Optional[Plan] = None

    def plan(
        self,
        decision: ReasoningDecision,
        world: Dict[str, Any],
    ) -> Plan:

        if decision.decision == "STOP":

            plan = Plan(
                status="READY",
                goal=None,
                reason=(
                    "El Reasoner determinó que "
                    "el robot debe detenerse."
                ),
            )

            plan.add_step(
                PlanStep(
                    action="STOP",
                )
            )

            self.last_plan = plan

            return plan

        if decision.decision == "MOVE":

            plan = Plan(
                status="READY",
                goal=None,
                reason=(
                    "Se planificó el movimiento "
                    "solicitado por el usuario."
                ),
            )

            plan.add_step(
                PlanStep(
                    action="MOVE",
                    direction=decision.direction,
                    parameters={
                        "distance": decision.metadata.get(
                            "requested_distance",
                            1,
                        )
                    },
                )
            )

            self.last_plan = plan

            return plan

        if decision.decision == "GOAL_REACHED":

            plan = Plan(
                status="COMPLETE",
                goal="GOAL",
                reason=(
                    "El robot ya se encuentra "
                    "en la meta."
                ),
            )

            self.last_plan = plan

            return plan

        if decision.decision == "GO_TO":

            return self._plan_to_goal(
                decision,
                world,
            )

        if decision.decision == "BLOCKED":

            plan = Plan(
                status="FAILED",
                reason=decision.reason,
                metadata={
                    "decision": decision.to_dict(),
                },
            )

            self.last_plan = plan

            return plan

        if decision.decision == "CLARIFY":

            plan = Plan(
                status="FAILED",
                reason=(
                    "No es posible crear un plan "
                    "sin una decisión suficientemente "
                    "definida."
                ),
                metadata={
                    "decision": decision.to_dict(),
                },
            )

            self.last_plan = plan

            return plan

        plan = Plan(
            status="FAILED",
            reason=(
                f"No existe una estrategia de "
                f"planificación para '{decision.decision}'."
            ),
        )

        self.last_plan = plan

        return plan

    def _plan_to_goal(
        self,
        decision: ReasoningDecision,
        world: Dict[str, Any],
    ) -> Plan:

        entities = world.get("entities", {})

        robot = entities.get("ROBOT")
        goal = entities.get("GOAL")

        if robot is None or goal is None:

            plan = Plan(
                status="FAILED",
                goal="GOAL",
                reason=(
                    "No se pudo localizar el robot "
                    "o la meta en el World Model."
                ),
            )

            self.last_plan = plan

            return plan

        start = robot.get(
            "properties",
            {},
        ).get("position")

        target = goal.get(
            "properties",
            {},
        ).get("position")

        if start is None or target is None:

            plan = Plan(
                status="FAILED",
                goal="GOAL",
                reason=(
                    "No se conoce la posición "
                    "del robot o de la meta."
                ),
            )

            self.last_plan = plan

            return plan

        path = self.path_planner.find_path(
            start=start,
            goal=target,
            world=world,
        )

        if not path:

            plan = Plan(
                status="FAILED",
                goal="GOAL",
                reason=(
                    "No existe un camino conocido "
                    "desde el robot hasta la meta."
                ),
                metadata={
                    "start": start,
                    "goal": target,
                },
            )

            self.last_plan = plan

            return plan

        directions = (
            self.path_planner.path_to_directions(path)
        )

        plan = Plan(
            status="READY",
            goal="GOAL",
            reason=(
                "Se encontró un camino válido "
                "hasta la meta."
            ),
            metadata={
                "start": start,
                "goal": target,
                "path": path,
            },
        )

        for direction in directions:

            plan.add_step(
                PlanStep(
                    action="MOVE",
                    direction=direction,
                    parameters={
                        "distance": 1,
                    },
                )
            )

        self.last_plan = plan

        return plan

    def get_last_plan(self):
        return self.last_plan

    def reset(self):
        self.last_plan = None