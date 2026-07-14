"""Send selected approach poses to NavigateToPose and handle results."""

from typing import Any, Dict

from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from robotino_emdb_interfaces.msg import RobotinoPolicyOutcome

from . import constants


class NavigationExecutionMixin:
    """NavigateToPose execution with stale-callback protection."""

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
                failure_reason=constants.FAILURE_NAVIGATION_FAILED,
                resume_exploration=self.resume_exploration_after_failure,
                navigation_result=RobotinoPolicyOutcome.NAV_FAILED,
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
                failure_reason=constants.FAILURE_NAVIGATION_FAILED,
                resume_exploration=self.resume_exploration_after_failure,
                navigation_result=RobotinoPolicyOutcome.NAV_FAILED,
            )
            return

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn("Nav2 approach goal was rejected.")
            self.finish_execution(
                generation,
                success=False,
                failure_reason=constants.FAILURE_GOAL_REJECTED,
                resume_exploration=self.resume_exploration_after_failure,
                navigation_result=RobotinoPolicyOutcome.NAV_FAILED,
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

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.stop_robot()
            self.enter_interaction_wait(generation)
            return

        if status == GoalStatus.STATUS_CANCELED:
            navigation_result = RobotinoPolicyOutcome.NAV_CANCELED
            failure_reason = constants.FAILURE_NAVIGATION_CANCELED
        elif status == GoalStatus.STATUS_ABORTED:
            navigation_result = RobotinoPolicyOutcome.NAV_FAILED
            failure_reason = constants.FAILURE_NAVIGATION_ABORTED
        else:
            navigation_result = RobotinoPolicyOutcome.NAV_FAILED
            failure_reason = constants.FAILURE_NAVIGATION_FAILED

        self.finish_execution(
            generation,
            success=False,
            failure_reason=failure_reason,
            resume_exploration=self.resume_exploration_after_failure,
            navigation_result=navigation_result,
        )
