from typing import List, Optional

from .context import ReasoningContext
from .decision import ReasoningDecision


class ReasoningRules:

    @staticmethod
    def evaluate(context: ReasoningContext) -> Optional[ReasoningDecision]:
        command = context.command

        if not command:
            return ReasoningDecision(
                decision="CLARIFY",
                reason="No se recibió un comando válido.",
                confidence=1.0,
            )

        intent = command.get("intent")
        action = command.get("action")

        if intent is None and action is None:
            return ReasoningDecision(
                decision="CLARIFY",
                reason="El comando no contiene información suficiente para razonar.",
                confidence=1.0,
            )

        # ---------------------------------------------------------
        # STOP
        # ---------------------------------------------------------

        if intent == "STOP" or action == "STOP":
            return ReasoningDecision(
                decision="STOP",
                reason="El usuario solicitó detener el robot.",
                confidence=1.0,
                action="STOP",
            )

        # ---------------------------------------------------------
        # MOVE
        # ---------------------------------------------------------

        if intent == "MOVE" or action == "MOVE":
            return ReasoningRules._evaluate_move(context)

        # ---------------------------------------------------------
        # GO_TO
        # ---------------------------------------------------------

        if intent == "GO_TO" or action == "GO_TO":
            return ReasoningRules._evaluate_go_to(context)

        # ---------------------------------------------------------
        # UNKNOWN
        # ---------------------------------------------------------

        return ReasoningDecision(
            decision="CLARIFY",
            reason=f"No existe una regla para la acción '{action}'.",
            confidence=0.5,
        )

    @staticmethod
    def _evaluate_move(
        context: ReasoningContext,
    ) -> ReasoningDecision:

        direction = context.get_command("direction")
        distance = context.get_command("distance", 1)

        if direction in (None, "NONE"):
            return ReasoningDecision(
                decision="CLARIFY",
                reason="El comando MOVE no especifica una dirección.",
                confidence=1.0,
            )

        robot_position = context.get_robot_position()

        if robot_position is None:
            return ReasoningDecision(
                decision="CLARIFY",
                reason="No se conoce la posición actual del robot.",
                confidence=1.0,
            )

        next_position = ReasoningRules._next_position(
            robot_position,
            direction,
        )

        if next_position is None:
            return ReasoningDecision(
                decision="CLARIFY",
                reason=f"La dirección '{direction}' no es válida.",
                confidence=1.0,
            )

        obstacles = context.get_obstacles()

        if next_position in obstacles:
            return ReasoningDecision(
                decision="BLOCKED",
                reason=(
                    f"La posición {next_position} está ocupada "
                    "por un obstáculo."
                ),
                confidence=1.0,
                action="MOVE",
                direction=direction,
                metadata={
                    "requested_distance": distance,
                    "next_position": next_position,
                },
            )

        environment = context.get_environment()

        width = environment.get("width")
        height = environment.get("height")

        if width is not None and height is not None:

            x, y = next_position

            if not (0 <= x < width and 0 <= y < height):
                return ReasoningDecision(
                    decision="BLOCKED",
                    reason=(
                        f"La posición {next_position} está "
                        "fuera de los límites del entorno."
                    ),
                    confidence=1.0,
                    action="MOVE",
                    direction=direction,
                    metadata={
                        "requested_distance": distance,
                        "next_position": next_position,
                    },
                )

        return ReasoningDecision(
            decision="MOVE",
            reason=(
                f"La dirección {direction} está disponible "
                f"desde la posición {robot_position}."
            ),
            confidence=1.0,
            action="MOVE",
            direction=direction,
            metadata={
                "requested_distance": distance,
                "next_position": next_position,
            },
        )

    @staticmethod
    def _evaluate_go_to(
        context: ReasoningContext,
    ) -> ReasoningDecision:

        target = context.get_command("target")

        if target in (None, "NONE"):
            return ReasoningDecision(
                decision="CLARIFY",
                reason="GO_TO no especifica un objetivo.",
                confidence=1.0,
            )

        robot_position = context.get_robot_position()

        if robot_position is None:
            return ReasoningDecision(
                decision="CLARIFY",
                reason="No se conoce la posición actual del robot.",
                confidence=1.0,
            )

        if target == "GOAL":

            goal_position = context.get_goal_position()

            if goal_position is None:
                return ReasoningDecision(
                    decision="CLARIFY",
                    reason="No existe una meta conocida en el mundo.",
                    confidence=1.0,
                )

            if robot_position == goal_position:
                return ReasoningDecision(
                    decision="GOAL_REACHED",
                    reason="El robot ya se encuentra en la meta.",
                    confidence=1.0,
                    action="GO_TO",
                    target="GOAL",
                )

            return ReasoningDecision(
                decision="GO_TO",
                reason=(
                    f"La meta está en {goal_position} y el robot "
                    f"está en {robot_position}."
                ),
                confidence=1.0,
                action="GO_TO",
                target="GOAL",
                metadata={
                    "robot_position": robot_position,
                    "goal_position": goal_position,
                },
            )

        return ReasoningDecision(
            decision="CLARIFY",
            reason=f"No se reconoce el objetivo '{target}'.",
            confidence=0.5,
        )

    @staticmethod
    def _next_position(
        position: List[int],
        direction: str,
    ):
        x, y = position

        movements = {
            "UP": (0, -1),
            "DOWN": (0, 1),
            "LEFT": (-1, 0),
            "RIGHT": (1, 0),
        }

        if direction not in movements:
            return None

        dx, dy = movements[direction]

        return [x + dx, y + dy]