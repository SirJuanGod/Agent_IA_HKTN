"""
Adaptador de entorno para ROS / ROS 2.

Implementa el requisito de mejoras.txt:
    "El modelo debe poder ejecutarse con cualquier base de simulador
    o en sistemas de puente como ROS/ROS 2."

Este módulo permite que SJG-Agent se conecte a:
    - ROS (Robot Operating System) mediante rospy
    - ROS 2 mediante rclpy
    - Modo simulado (mock) cuando ROS no está disponible.

Uso con ROS 2 real:
--------------------------------------------------------------
    from rbj.environment.ros_adapter import ROS2EnvironmentAdapter

    adapter = ROS2EnvironmentAdapter(
        node_name="sjg_agent",
        cmd_vel_topic="/cmd_vel",
        odom_topic="/odom",
        goal_topic="/goal_pose",
    )
    env = EnvironmentAdapter(adapter)

Uso en modo simulado (sin ROS):
--------------------------------------------------------------
    adapter = ROS2EnvironmentAdapter(mock=True)
    env = EnvironmentAdapter(adapter)
--------------------------------------------------------------
"""

import warnings
from typing import Any, Dict, Optional

from rbj.environment.base import Environment


# ============================================================
# Comprobación de disponibilidad de ROS
# ============================================================

def _check_ros2() -> bool:
    try:
        import rclpy  # noqa: F401
        return True
    except ImportError:
        return False


def _check_ros1() -> bool:
    try:
        import rospy  # noqa: F401
        return True
    except ImportError:
        return False


ROS2_AVAILABLE = _check_ros2()
ROS1_AVAILABLE = _check_ros1()


# ============================================================
# Adaptador ROS 2
# ============================================================

class ROS2EnvironmentAdapter(Environment):
    """
    Adaptador entre SJG-Agent y un entorno ROS 2.

    Cuando `mock=True`, opera en modo simulado sin necesidad
    de una instalación de ROS.

    Parámetros
    ----------
    node_name : str
        Nombre del nodo ROS 2.
    cmd_vel_topic : str
        Topic de publicación de velocidades (geometry_msgs/Twist).
    odom_topic : str
        Topic de suscripción de odometría (nav_msgs/Odometry).
    goal_topic : str
        Topic de publicación de metas (geometry_msgs/PoseStamped).
    mock : bool
        Si True, opera en modo simulado sin ROS.
    """

    def __init__(
        self,
        node_name: str = "sjg_agent_node",
        cmd_vel_topic: str = "/cmd_vel",
        odom_topic: str = "/odom",
        goal_topic: str = "/goal_pose",
        mock: bool = False,
    ):
        self._mock = mock
        self._done = False
        self._position = [0, 0]

        self._cmd_vel_topic = cmd_vel_topic
        self._odom_topic = odom_topic
        self._goal_topic = goal_topic

        if mock:
            warnings.warn(
                "ROS2EnvironmentAdapter operando en modo SIMULADO (mock=True). "
                "No se enviarán comandos reales a ROS 2.",
                stacklevel=2,
            )
            return

        if not ROS2_AVAILABLE:
            raise ImportError(
                "rclpy no está instalado. Instala ROS 2 o usa mock=True "
                "para operar en modo simulado.\n"
                "Instalación: https://docs.ros.org/en/humble/Installation.html"
            )

        import rclpy
        from rclpy.node import Node

        if not rclpy.ok():
            rclpy.init()

        self._node = Node(node_name)

        # Importaciones diferidas de tipos de mensajes
        from geometry_msgs.msg import Twist, PoseStamped
        from nav_msgs.msg import Odometry

        self._twist_msg = Twist

        self._publisher = self._node.create_publisher(
            Twist,
            cmd_vel_topic,
            10,
        )

        self._goal_publisher = self._node.create_publisher(
            PoseStamped,
            goal_topic,
            10,
        )

        self._last_odom: Optional[Dict] = None

        self._odom_sub = self._node.create_subscription(
            Odometry,
            odom_topic,
            self._odom_callback,
            10,
        )

    # ----------------------------------------------------------
    # Callbacks
    # ----------------------------------------------------------

    def _odom_callback(self, msg) -> None:
        """Actualiza la posición del robot desde odometría."""

        self._last_odom = {
            "x": msg.pose.pose.position.x,
            "y": msg.pose.pose.position.y,
            "z": msg.pose.pose.position.z,
        }

    # ----------------------------------------------------------
    # Environment interface
    # ----------------------------------------------------------

    def reset(self) -> Dict[str, Any]:
        """Reinicia el entorno."""

        self._done = False
        self._position = [0, 0]

        if not self._mock:
            self._send_velocity(0.0, 0.0)

        return self._build_observation()

    def observe(self) -> Dict[str, Any]:
        """Obtiene la observación actual del entorno."""

        if not self._mock and self._last_odom:
            self._position = [
                round(self._last_odom["x"]),
                round(self._last_odom["y"]),
            ]

        return self._build_observation()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta una acción en el entorno.

        Convierte las acciones de SJG-Agent en comandos ROS 2.

        Acciones soportadas:
            {"type": "MOVE", "direction": "UP"}   → linear.x = +vel
            {"type": "MOVE", "direction": "DOWN"}  → linear.x = -vel
            {"type": "MOVE", "direction": "LEFT"}  → angular.z = +vel
            {"type": "MOVE", "direction": "RIGHT"} → angular.z = -vel
            {"type": "STOP"}                        → velocidad 0
        """

        action_type = action.get("type") or action.get("action", "STOP")
        direction   = action.get("direction", "NONE")

        if action_type == "STOP":
            if not self._mock:
                self._send_velocity(0.0, 0.0)

        elif action_type == "MOVE":
            linear, angular = self._direction_to_velocity(direction)

            if not self._mock:
                self._send_velocity(linear, angular)

            # En modo mock actualizamos posición internamente
            dx, dy = self._direction_to_delta(direction)
            self._position[0] += dx
            self._position[1] += dy

        elif action_type == "GO_TO":
            # GO_TO publica una meta en /goal_pose para que
            # el stack de navegación de ROS 2 (Nav2) planifique
            if not self._mock:
                self._publish_goal(action.get("target_position", [0, 0]))

        return self._build_observation()

    def is_done(self) -> bool:
        """El agente decide cuándo termina el episodio externamente."""
        return self._done

    def close(self) -> None:
        """Libera los recursos del nodo ROS 2."""

        if not self._mock and ROS2_AVAILABLE:
            try:
                import rclpy
                self._node.destroy_node()
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

    def _send_velocity(
        self,
        linear: float,
        angular: float,
    ) -> None:
        """Publica un mensaje Twist en /cmd_vel."""

        msg = self._twist_msg()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self._publisher.publish(msg)

    def _publish_goal(self, position) -> None:
        """Publica una meta en /goal_pose."""

        from geometry_msgs.msg import PoseStamped

        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0
        self._goal_publisher.publish(msg)

    @staticmethod
    def _direction_to_velocity(direction: str):
        """Convierte dirección en (linear, angular)."""

        mapping = {
            "UP":    (0.3,  0.0),
            "DOWN":  (-0.3, 0.0),
            "LEFT":  (0.0,  0.5),
            "RIGHT": (0.0, -0.5),
        }
        return mapping.get(direction, (0.0, 0.0))

    @staticmethod
    def _direction_to_delta(direction: str):
        """Convierte dirección en (dx, dy) para el mock."""

        mapping = {
            "UP":    (0, -1),
            "DOWN":  (0,  1),
            "LEFT":  (-1, 0),
            "RIGHT": (1,  0),
        }
        return mapping.get(direction, (0, 0))

    def _build_observation(self) -> Dict[str, Any]:
        """Construye el dict de observación estándar de SJG-Agent."""

        return {
            "environment": {
                "type": "ROS2",
                "ros2_available": ROS2_AVAILABLE,
                "mock": self._mock,
            },
            "entities": {
                "ROBOT": {
                    "entity_id": "ROBOT",
                    "entity_type": "ROBOT",
                    "properties": {
                        "position": list(self._position),
                    },
                },
            },
        }


# ============================================================
# Adaptador ROS 1 (legacy)
# ============================================================

class ROS1EnvironmentAdapter(Environment):
    """
    Adaptador entre SJG-Agent y un entorno ROS 1 (rospy).

    Usar preferentemente ROS2EnvironmentAdapter para nuevos proyectos.

    Parámetros
    ----------
    node_name : str
        Nombre del nodo ROS.
    mock : bool
        Si True, opera en modo simulado sin ROS.
    """

    def __init__(
        self,
        node_name: str = "sjg_agent_node",
        cmd_vel_topic: str = "/cmd_vel",
        mock: bool = False,
    ):
        self._mock = mock
        self._done = False
        self._position = [0, 0]

        if mock:
            warnings.warn(
                "ROS1EnvironmentAdapter operando en modo SIMULADO (mock=True).",
                stacklevel=2,
            )
            return

        if not ROS1_AVAILABLE:
            raise ImportError(
                "rospy no está instalado. Instala ROS 1 o usa mock=True."
            )

        import rospy
        from geometry_msgs.msg import Twist

        rospy.init_node(node_name, anonymous=True)

        self._publisher = rospy.Publisher(
            cmd_vel_topic,
            Twist,
            queue_size=10,
        )

        self._twist_msg = Twist

    def reset(self) -> Dict[str, Any]:
        self._done = False
        self._position = [0, 0]
        return self._build_observation()

    def observe(self) -> Dict[str, Any]:
        return self._build_observation()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        action_type = action.get("type") or action.get("action", "STOP")
        direction   = action.get("direction", "NONE")

        if action_type == "MOVE":
            dx, dy = ROS2EnvironmentAdapter._direction_to_delta(direction)
            self._position[0] += dx
            self._position[1] += dy

            if not self._mock:
                linear, angular = ROS2EnvironmentAdapter._direction_to_velocity(
                    direction
                )
                msg = self._twist_msg()
                msg.linear.x = float(linear)
                msg.angular.z = float(angular)
                self._publisher.publish(msg)

        return self._build_observation()

    def is_done(self) -> bool:
        return self._done

    def close(self) -> None:
        if not self._mock and ROS1_AVAILABLE:
            try:
                import rospy
                rospy.signal_shutdown("SJG-Agent cerrando.")
            except Exception:
                pass

    def _build_observation(self) -> Dict[str, Any]:
        return {
            "environment": {
                "type": "ROS1",
                "ros1_available": ROS1_AVAILABLE,
                "mock": self._mock,
            },
            "entities": {
                "ROBOT": {
                    "entity_id": "ROBOT",
                    "entity_type": "ROBOT",
                    "properties": {
                        "position": list(self._position),
                    },
                },
            },
        }
