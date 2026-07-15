#!/usr/bin/env python3
"""Robotino-specific implementations of official e-MDB node roles.

Every class deliberately registers with the canonical GII type so the LTM and
MainLoop see PNode, CNode, Goal, and WorldModel—not Python subclass names.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy
import rclpy
from cognitive_nodes.cnode import CNode
from cognitive_nodes.goal import Goal
from cognitive_nodes.pnode import PNode
from cognitive_node_interfaces.msg import Activation
from cognitive_node_interfaces.srv import GetActivation
from core.cognitive_node import CognitiveNode
from core.service_client import ServiceClientAsync
from core.utils import perception_dict_to_msg


class RobotinoSeededPNode(PNode):
    """P-Node with a safe initial context and normal e-MDB learning services.

    Seeded P-Nodes use a deterministic prior context for initial autonomous
    behavior.  They still expose the official add-point/add-antipoint services,
    so MainLoop episodes are recorded in their normal point space.  When
    the configured seed context is empty (as for P-Nodes created later by MainLoop), activation
    falls back completely to the standard learned PointBasedSpace behavior.
    """

    SUPPORTED_CONTEXTS = {
        "explore",
        "search_energy",
        "return_energy",
        "go_goal",
        "wait_safe",
    }

    def __init__(
        self,
        name: str = "pnode",
        class_name: str = "cognitive_nodes.pnode.PNode",
        context: str = "",
        context_sensor: str = "robotino_context",
        space_class: str = "cognitive_nodes.space.PointBasedSpace",
        history_size: int = 100,
        **params,
    ) -> None:
        self.seed_context = str(context).strip().lower()
        self.context_sensor = str(context_sensor)
        super().__init__(
            name=name,
            class_name="cognitive_nodes.pnode.PNode",
            space_class=space_class,
            history_size=history_size,
            **params,
        )
        if self.seed_context and self.seed_context not in self.SUPPORTED_CONTEXTS:
            raise ValueError(
                f"Unsupported Robotino P-Node context: {self.seed_context}"
            )
        self.get_logger().info(
            f"Robotino P-Node '{name}' initialized with context="
            f"{self.seed_context or '<learned>'}"
        )

    @staticmethod
    def flag(data: Dict[str, float], key: str) -> bool:
        return float(data.get(key, 0.0)) >= 0.5

    def context_activation(self, data: Dict[str, float]) -> float:
        valid = self.flag(data, "state_valid")
        recovery = self.flag(data, "energy_recovery_mode")
        bank_worthy = self.flag(data, "bank_worthy")
        mapping_complete = self.flag(data, "mapping_complete")
        frontiers = self.flag(data, "frontiers_available")
        goal_known = self.flag(data, "goal_known")
        goal_satisfied = self.flag(data, "goal_satisfied")

        explore = (
            valid
            and not recovery
            and not mapping_complete
            and frontiers
            and not goal_satisfied
        )
        search_energy = (
            valid
            and recovery
            and not bank_worthy
            and frontiers
            and not goal_satisfied
        )
        return_energy = (
            valid
            and recovery
            and bank_worthy
            and not goal_satisfied
        )
        go_goal = (
            valid
            and not recovery
            and mapping_complete
            and goal_known
            and not goal_satisfied
        )
        wait_safe = not (
            explore or search_energy or return_energy or go_goal
        )

        values = {
            "explore": explore,
            "search_energy": search_energy,
            "return_energy": return_energy,
            "go_goal": go_goal,
            "wait_safe": wait_safe,
        }
        return 1.0 if values[self.seed_context] else 0.0

    def add_point_callback(self, request, response):
        """Acknowledge learning updates safely for fixed seeded contexts.

        The deterministic seed context, rather than PointBasedSpace, owns the
        activation of these bootstrap P-Nodes. The stock MainLoop nevertheless
        sends points and antipoints after every episode. Passing those updates
        into PointBasedSpace can block or fail on the first antipoint and cannot
        affect this node's activation anyway.

        Dynamically created P-Nodes have an empty ``seed_context`` and retain the
        original e-MDB learning behavior.
        """
        if not self.seed_context:
            return super().add_point_callback(request, response)

        confidence = float(request.confidence)
        update_kind = "point" if confidence > 0.0 else "antipoint"
        self.get_logger().info(
            f"Acknowledged {update_kind} for fixed seeded context "
            f"'{self.seed_context}' (confidence={confidence:.1f}); "
            "context activation remains rule-seeded."
        )
        response.added = True
        return response

    def add_points_callback(self, request, response):
        """Batch equivalent of :meth:`add_point_callback`."""
        if not self.seed_context:
            return super().add_points_callback(request, response)

        positives = sum(
            1 for confidence in request.confidences if float(confidence) > 0.0
        )
        antipoints = len(request.confidences) - positives
        self.get_logger().info(
            f"Acknowledged batch update for fixed seeded context "
            f"'{self.seed_context}': points={positives}, antipoints={antipoints}; "
            "context activation remains rule-seeded."
        )
        response.added = bool(request.points)
        return response

    def calculate_activation(self, perception=None, activation_list=None):
        # P-Nodes created dynamically by MainLoop have no seed context and use
        # the official learned-space calculation unchanged.
        if not self.seed_context:
            return super().calculate_activation(
                perception=perception,
                activation_list=activation_list,
            )

        if activation_list is not None:
            perception = {
                sensor: values["data"]
                for sensor, values in activation_list.items()
            }
            timestamps = [
                values.get("timestamp")
                for values in activation_list.values()
                if values.get("timestamp") is not None
            ]
            for values in activation_list.values():
                values["updated"] = False
        else:
            timestamps = []

        sensor_values = (perception or {}).get(self.context_sensor, [])
        data = sensor_values[0] if sensor_values else {}
        self.activation.activation = self.context_activation(data)
        self.activation.timestamp = (
            max(timestamps, key=lambda stamp: stamp.nanoseconds).to_msg()
            if timestamps
            else self.get_clock().now().to_msg()
        )
        return self.activation


class RobotinoCNode(CNode):
    """Correct C-Node product activation for the installed GII release."""

    def __init__(
        self,
        name: str = "cnode",
        class_name: str = "cognitive_nodes.cnode.CNode",
        **params,
    ) -> None:
        super().__init__(
            name=name,
            class_name="cognitive_nodes.cnode.CNode",
            **params,
        )

    async def calculate_activation(self, perception=None, activation_list=None):
        if activation_list is not None:
            self.calculate_activation_prod(activation_list)
            return self.activation

        node_activations = []
        for neighbor in self.neighbors:
            if neighbor["node_type"] == "Policy":
                continue
            name = neighbor["name"]
            service_name = f"cognitive_node/{name}/get_activation"
            if service_name not in self.node_clients:
                self.node_clients[service_name] = ServiceClientAsync(
                    self,
                    GetActivation,
                    service_name,
                    self.cbgroup_client,
                )
            result = await self.node_clients[service_name].send_request_async(
                perception=perception_dict_to_msg(perception or {})
            )
            node_activations.append(float(result.activation))

        self.activation.activation = (
            float(numpy.prod(node_activations))
            if node_activations
            else 0.0
        )
        self.activation.timestamp = self.get_clock().now().to_msg()
        return self.activation


class RobotinoStaticWorldModel(CognitiveNode):
    """Torch-free static WorldModel for the Robotino foraging domain."""

    def __init__(
        self,
        name: str = "robotino_world",
        class_name: str = "cognitive_nodes.world_model.WorldModel",
        activation_value: float = 1.0,
        **params,
    ) -> None:
        super().__init__(
            name,
            "cognitive_nodes.world_model.WorldModel",
            **params,
        )
        self.activation_value = max(0.0, min(1.0, float(activation_value)))
        self.activation.activation = self.activation_value
        self.activation.timestamp = self.get_clock().now().to_msg()

    def calculate_activation(self, perception=None, activation_list=None):
        self.activation.activation = self.activation_value
        self.activation.timestamp = self.get_clock().now().to_msg()
        return self.activation


class RobotinoSafetyGoal(Goal):
    """Always available, zero-reward Goal used only by wait_safe."""

    def __init__(
        self,
        name: str = "safe_operation_goal",
        class_name: str = "cognitive_nodes.goal.Goal",
        activation_value: float = 1.0,
        **params,
    ) -> None:
        super().__init__(
            name=name,
            class_name="cognitive_nodes.goal.Goal",
            **params,
        )
        self.activation_value = max(0.0, min(1.0, float(activation_value)))
        self.activation.activation = self.activation_value
        self.activation.timestamp = self.get_clock().now().to_msg()

    def calculate_activation(self, perception=None, activation_list=None):
        self.activation.activation = self.activation_value
        self.activation.timestamp = self.get_clock().now().to_msg()
        return self.activation

    def get_reward_callback(self, request, response):
        """Return a fresh zero reward without entering the base retry loop.

        ``safe_operation_goal`` is a structural fallback goal, not a learned
        objective.  The stock Goal callback derives ``updated`` from message
        timestamps.  During MainLoop startup that response can arrive as stale
        or fail to reach the temporary service client, leaving
        ``get_goals_reward()`` retrying forever before iteration 1.

        This goal always has a valid, immediately available zero reward.
        """
        response.reward = 0.0
        response.updated = True
        self.get_logger().info(
            "Obtaining reward from safe_operation_goal => 0.0 "
            "(structural fallback)"
        )
        return response

    def get_reward(self, old_perception=None, perception=None):
        return 0.0, self.get_clock().now().to_msg()


def main(args=None) -> None:
    # Standalone smoke test for the static WorldModel only.  In normal use the
    # Commander creates all classes from the experiment YAML.
    rclpy.init(args=args)
    node = RobotinoStaticWorldModel()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()