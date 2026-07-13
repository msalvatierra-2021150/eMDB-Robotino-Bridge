"""Navigation execution: send the selected approach pose to NavigateToPose
and handle its result.

Stale Nav2 results (from a cancelled or superseded execution generation)
are dropped here so they can never be attributed to a newer policy.
"""

from typing import Any, Dict

from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose


class NavigationExecutionMixin:
    """Requires from the host class:

    Attributes: nav2_client, active_navigation_goal_handle, last_goal_time.
    Methods: get_execution(), finish_execution(), enter_interaction_wait(),
        stop_robot().
    Constants: STAGE_PLANNING, STAGE_NAVIGATING.
    """

    def send_navigation_goal(
        self,
        generation: int,
        selected: Dict[str, Any],
    ) -> None:
        context = self.get_execution(generation, self.STAGE_PLANNING)
        if context is None:
            return

        if not self.nav2_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error(
                "Nav2 NavigateToPose action server is unavailable."
            )
            self.finish_execution(
                generation,
                success=False,
                status="nav2_server_unavailable",
                resume_exploration=self.resume_exploration_after_failure,
            )
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = selected["pose"]
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        context["stage"] = self.STAGE_NAVIGATING
        self.last_goal_time = self.get_clock().now()

        send_future = self.nav2_client.send_goal_async(goal_msg)
        send_future.add_done_callback(
            lambda future, g=generation:
            self.nav2_goal_response_callback(future, g)
        )

    def nav2_goal_response_callback(self, future, generation: int) -> None:
        context = self.get_execution(generation, self.STAGE_NAVIGATING)
        if context is None:
            return

        try:
            goal_handle = future.result()
        except Exception as ex:  # noqa: BLE001
            self.get_logger().error(f"Failed to send Nav2 goal: {ex}")
            self.finish_execution(
                generation,
                success=False,
                status="nav2_goal_send_failed",
                resume_exploration=self.resume_exploration_after_failure,
            )
            return

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn("Nav2 approach goal was rejected.")
            self.finish_execution(
                generation,
                success=False,
                status="nav2_goal_rejected",
                resume_exploration=self.resume_exploration_after_failure,
            )
            return

        self.active_navigation_goal_handle = goal_handle
        self.get_logger().info("Nav2 approach goal accepted.")

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda result, g=generation:
            self.nav2_result_callback(result, g)
        )

    def nav2_result_callback(self, future, generation: int) -> None:
        context = self.get_execution(generation, self.STAGE_NAVIGATING)
        if context is None:
            # This is the critical stale-callback protection. A result from a
            # cancelled/older policy is never applied to the current policy.
            self.get_logger().debug(
                f"Ignoring stale Nav2 result for generation {generation}."
            )
            return

        self.active_navigation_goal_handle = None

        try:
            wrapped_result = future.result()
            status = int(wrapped_result.status)
        except Exception as ex:  # noqa: BLE001
            self.get_logger().error(
                f"Failed to receive Nav2 result: {ex}"
            )
            status = -1

        if status != GoalStatus.STATUS_SUCCEEDED:
            self.finish_execution(
                generation,
                success=False,
                status=f"nav2_goal_finished_status_{status}",
                resume_exploration=self.resume_exploration_after_failure,
            )
            return

        self.stop_robot()
        self.enter_interaction_wait(generation)