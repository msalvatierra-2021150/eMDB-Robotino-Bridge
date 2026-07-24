"""Mapped-space wandering integrated with Robotino policy execution.

The generic frontier_exploration submodule remains unchanged. Once the existing
``/frontier_exploration/mapping_complete`` publisher reports ``True``, the same
free-space sampler supports both adequate-energy ``wander_mapped_space`` and
low-energy ``search_for_energy``. Each selection performs one blocking,
outcome-producing Nav2 action before control returns to eMDB.

AprilTag observations continue through the existing bridge and foraging-memory
node while this action runs, so first sightings create memory and repeated
sightings update the same memory record without coupling perception to motion.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Sequence, Tuple

from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool, Float32
from robotino_emdb_interfaces.msg import (
    RobotinoPolicyOutcome,
    RobotinoSelectedPolicy,
)

from . import constants


class MappedWanderingMixin:
    """Sample mapped free space for novelty or energy-motivated search."""

    WANDER_PURPOSE_NOVELTY = "novelty"
    WANDER_PURPOSE_ENERGY_SEARCH = "energy_search"

    def execute_wander_mapped_space(
        self,
        policy: RobotinoSelectedPolicy,
    ) -> None:
        """Start one adequate-energy semantic wandering action."""
        self.execute_mapped_wandering(
            policy,
            purpose=self.WANDER_PURPOSE_NOVELTY,
        )

    def execute_search_for_energy_mapped_space(
        self,
        policy: RobotinoSelectedPolicy,
    ) -> None:
        """Search mapped space when energy is low and no useful bank is known."""
        self.execute_mapped_wandering(
            policy,
            purpose=self.WANDER_PURPOSE_ENERGY_SEARCH,
        )

    def execute_mapped_wandering(
        self,
        policy: RobotinoSelectedPolicy,
        purpose: str,
    ) -> None:
        """Start one eMDB-controlled, outcome-producing wandering action."""
        if self.is_same_policy_already_active(policy):
            self.get_logger().debug(
                "Repeated mapped-wandering policy ignored; execution is active."
            )
            return

        if self.execution is not None:
            self.cancel_current_execution(
                reason=f"preempted_by_new_{purpose}_wandering_policy",
                publish_outcome=True,
            )

        # Frontier and mapped wandering must never own Nav2 simultaneously.
        self.set_exploration_enabled(False)

        if not bool(policy.use_nav2):
            self.publish_simple_outcome(
                policy,
                success=False,
                failure_reason=constants.FAILURE_POLICY_DOES_NOT_USE_NAV2,
            )
            return

        if not self.enable_nav2_execution:
            self.get_logger().warn(
                "Mapped wandering requires enable_nav2_execution:=true."
            )
            self.publish_simple_outcome(
                policy,
                success=False,
                failure_reason=constants.FAILURE_EXECUTION_DISABLED,
            )
            return

        if not self.mapping_complete:
            self.get_logger().warn(
                "Mapped wandering requested before mapping_complete=true."
            )
            self.publish_simple_outcome(
                policy,
                success=False,
                failure_reason=constants.FAILURE_POLICY_PREEMPTED,
            )
            return

        state = self.latest_foraging_state
        if state is None:
            self.publish_simple_outcome(
                policy,
                success=False,
                failure_reason=constants.FAILURE_ENERGY_DATA_UNAVAILABLE,
            )
            return

        if not self.wander_state_allows_motion(state, purpose):
            self.get_logger().info(
                f"Mapped {purpose} wandering withheld by current state."
            )
            self.publish_simple_outcome(
                policy,
                success=False,
                failure_reason=constants.FAILURE_POLICY_PREEMPTED,
            )
            return

        if not self.goal_interval_ok():
            self.get_logger().debug(
                "Mapped wandering delayed by minimum Nav2 goal interval."
            )
            self.publish_simple_outcome(
                policy,
                success=False,
                failure_reason=constants.FAILURE_POLICY_PREEMPTED,
            )
            return

        robot_position = self.get_robot_position_from_tf()
        if robot_position is None:
            self.publish_simple_outcome(
                policy,
                success=False,
                failure_reason=constants.FAILURE_NO_ROBOT_POSE,
            )
            return

        candidates = self.build_wander_candidates(*robot_position)
        if not candidates:
            self.get_logger().warn(
                f"No clearance-safe mapped-space {purpose} candidates found."
            )
            self.publish_simple_outcome(
                policy,
                success=False,
                failure_reason=constants.FAILURE_TARGET_UNREACHABLE,
                navigation_result=RobotinoPolicyOutcome.NAV_FAILED,
            )
            return

        self.execution_generation += 1
        generation = self.execution_generation
        self.execution = {
            "generation": generation,
            "policy": copy.deepcopy(policy),
            "stage": self.STAGE_PLANNING,
            "navigation_mode": "wander",
            "wander_purpose": purpose,
            "target_x": 0.0,
            "target_y": 0.0,
            "energy_before": self.get_current_energy(),
            "reward_before": self.get_current_reward(),
            "arrival_energy": None,
            "arrival_reward": None,
            "interaction_deadline_ns": None,
            "selected_candidate_name": None,
        }
        self.candidate_plan_candidates = candidates
        self.candidate_plan_index = 0

        self.get_logger().info(
            f"Checking mapped-space {purpose} candidates with "
            f"ComputePathToPose: candidates={len(candidates)}, "
            f"energy={float(state.robot_energy):.3f}."
        )

        if not self.compute_path_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error(
                "Nav2 ComputePathToPose action server is unavailable."
            )
            self.finish_execution(
                generation,
                success=False,
                failure_reason=constants.FAILURE_PATH_UNAVAILABLE,
                resume_exploration=False,
                navigation_result=RobotinoPolicyOutcome.NAV_FAILED,
            )
            return

        self.request_next_candidate_path(generation)

    def energy_bank_is_actionable(self, state) -> bool:
        """Trust the memory node's centralized bank evaluation."""
        best_tag_id = int(
            getattr(state, "best_energy_tag_id", -1)
        )
        best_score = max(
            0.0,
            float(getattr(state, "best_energy_score", 0.0)),
        )

        return best_tag_id >= 0 and best_score > 0.0

    def wander_state_allows_motion(self, state, purpose: str) -> bool:
        """Check the state constraints for the requested wandering purpose."""
        if not bool(getattr(state, "valid", False)):
            return False
        if bool(getattr(state, "goal_satisfied", False)):
            return False

        if purpose == self.WANDER_PURPOSE_ENERGY_SEARCH:
            # Low energy is the reason for moving; never apply the normal
            # adequate-energy threshold. Stop when energy has recovered or
            # memory offers a worthy bank that can activate return_to_energy.
            if float(state.robot_energy) >= self.resume_energy_threshold:
                return False
            return not self.energy_bank_is_actionable(state)

        if float(state.robot_energy) <= self.wander_energy_threshold:
            return False
        # During adequate-energy wandering, a remembered mission goal wins.
        return not bool(getattr(state, "goal_known", False))

    def handle_wander_state_change(self, state) -> None:
        """Preempt mapped wandering when a higher-priority context appears."""
        context = self.execution
        if context is None or context.get("navigation_mode") != "wander":
            return

        purpose = str(
            context.get("wander_purpose", self.WANDER_PURPOSE_NOVELTY)
        )
        reason = None

        if not self.mapping_complete:
            reason = "mapping_no_longer_complete"
        elif not bool(getattr(state, "valid", False)):
            reason = "foraging_state_invalid"
        elif bool(getattr(state, "goal_satisfied", False)):
            reason = "mission_goal_satisfied"
        elif purpose == self.WANDER_PURPOSE_ENERGY_SEARCH:
            if float(state.robot_energy) >= self.resume_energy_threshold:
                reason = "energy_recovered_during_search"
            elif self.energy_bank_is_actionable(state):
                reason = "worthy_energy_bank_discovered"
        elif float(state.robot_energy) <= self.wander_energy_threshold:
            reason = "energy_reached_wander_threshold"
        elif bool(getattr(state, "goal_known", False)):
            reason = "remembered_goal_now_has_priority"

        if reason is not None:
            self.cancel_current_execution(reason=reason, publish_outcome=True)

    def update_mapping_complete(self, value: bool, source: str) -> None:
        """Apply one mapping-completion update and preempt if it resets."""
        previous = self.mapping_complete
        self.mapping_complete = bool(value)

        if self.mapping_complete and self.frontier_exploration_enabled:
            self.get_logger().info(
                "Mapping completed; disabling persistent frontier motion "
                "before mapped-space policy execution."
            )
            self.set_exploration_enabled(False)

        if previous != self.mapping_complete:
            self.get_logger().info(
                "Mapping completion changed to "
                f"{self.mapping_complete} from {source}."
            )

        state = self.latest_foraging_state
        if state is not None:
            self.handle_wander_state_change(state)

    def mapping_complete_callback(self, msg: Bool) -> None:
        """Track the existing frontier completion signal directly."""
        self.mapping_complete_signal_received = True
        self.update_mapping_complete(bool(msg.data), "mapping_complete topic")

    def exploration_satisfaction_callback(self, msg: Float32) -> None:
        """Fallback to the current cognitive-signals adapter when necessary.

        RobotinoContextPerception already infers mapping completion from this
        recurrent satisfaction signal. The executor uses it only until the
        direct Bool topic has been observed, avoiding conflicting authorities.
        """
        if self.mapping_complete_signal_received:
            return
        self.update_mapping_complete(
            float(msg.data) >= 0.999,
            "exploration satisfaction fallback",
        )

    def map_callback(self, msg: OccupancyGrid) -> None:
        self.latest_map = msg

    def build_wander_candidates(
        self,
        robot_x: float,
        robot_y: float,
    ) -> List[Dict[str, Any]]:
        """Sample known free cells and return candidates for path validation."""
        grid = self.latest_map
        if grid is None:
            self.get_logger().warn(
                f"No OccupancyGrid received on {self.map_topic}."
            )
            return []

        width = int(grid.info.width)
        height = int(grid.info.height)
        resolution = float(grid.info.resolution)
        data: Sequence[int] = grid.data

        if (
            width <= 0
            or height <= 0
            or resolution <= 0.0
            or len(data) != width * height
        ):
            self.get_logger().warn("Received an invalid OccupancyGrid.")
            return []

        sampled_indices = set()
        scored_cells: List[Tuple[float, float, float]] = []
        total_cells = width * height
        max_attempts = min(total_cells, self.wander_candidate_samples * 10)

        for _ in range(max_attempts):
            if len(scored_cells) >= self.wander_candidate_samples:
                break

            index = self.wander_random.randrange(total_cells)
            if index in sampled_indices:
                continue
            sampled_indices.add(index)

            row, col = divmod(index, width)
            if not self.wander_cell_has_clearance(
                data,
                width,
                height,
                row,
                col,
                resolution,
            ):
                continue

            x, y = self.wander_cell_center_world(grid, row, col)
            distance_from_robot = math.hypot(x - robot_x, y - robot_y)
            if (
                distance_from_robot < self.wander_min_distance_m
                or distance_from_robot > self.wander_max_distance_m
            ):
                continue

            distance_from_recent = self.distance_to_recent_wander_goals(x, y)
            if distance_from_recent < self.wander_recent_goal_radius_m:
                continue

            # Prefer spatial novelty, with a smaller reward for useful travel.
            score = distance_from_recent + 0.20 * distance_from_robot
            scored_cells.append((score, x, y))

        scored_cells.sort(key=lambda item: item[0], reverse=True)
        candidates: List[Dict[str, Any]] = []

        for index, (score, x, y) in enumerate(
            scored_cells[: self.wander_path_checks]
        ):
            yaw = math.atan2(y - robot_y, x - robot_x)
            candidates.append(
                {
                    "name": f"mapped_wander_{index}",
                    "x": x,
                    "y": y,
                    "yaw": yaw,
                    "pose": self.make_pose_stamped(x, y, yaw),
                    "valid": False,
                    "path_length": math.inf,
                    "wander_score": score,
                }
            )

        return candidates

    def wander_cell_has_clearance(
        self,
        data: Sequence[int],
        width: int,
        height: int,
        row: int,
        col: int,
        resolution: float,
    ) -> bool:
        radius_cells = int(math.ceil(self.wander_clearance_m / resolution))
        row_min = row - radius_cells
        row_max = row + radius_cells
        col_min = col - radius_cells
        col_max = col + radius_cells

        if (
            row_min < 0
            or col_min < 0
            or row_max >= height
            or col_max >= width
        ):
            return False

        radius_squared = radius_cells * radius_cells
        for check_row in range(row_min, row_max + 1):
            dr = check_row - row
            for check_col in range(col_min, col_max + 1):
                dc = check_col - col
                if dr * dr + dc * dc > radius_squared:
                    continue

                occupancy = int(data[check_row * width + check_col])
                # Unknown or occupied/inflated cells are never selected.
                if occupancy < 0 or occupancy > self.wander_free_threshold:
                    return False

        return True

    @staticmethod
    def wander_cell_center_world(
        grid: OccupancyGrid,
        row: int,
        col: int,
    ) -> Tuple[float, float]:
        resolution = float(grid.info.resolution)
        local_x = (float(col) + 0.5) * resolution
        local_y = (float(row) + 0.5) * resolution
        origin = grid.info.origin
        yaw = MappedWanderingMixin.quaternion_to_yaw(
            float(origin.orientation.x),
            float(origin.orientation.y),
            float(origin.orientation.z),
            float(origin.orientation.w),
        )

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        world_x = (
            float(origin.position.x)
            + cos_yaw * local_x
            - sin_yaw * local_y
        )
        world_y = (
            float(origin.position.y)
            + sin_yaw * local_x
            + cos_yaw * local_y
        )
        return world_x, world_y

    @staticmethod
    def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
        sin_yaw = 2.0 * (w * z + x * y)
        cos_yaw = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(sin_yaw, cos_yaw)

    def distance_to_recent_wander_goals(self, x: float, y: float) -> float:
        if not self.recent_wander_goals:
            return self.wander_max_distance_m
        return min(
            math.hypot(x - old_x, y - old_y)
            for old_x, old_y in self.recent_wander_goals
        )

    def register_selected_wander_candidate(
        self,
        candidate: Dict[str, Any],
    ) -> None:
        """Remember and publish the concrete goal selected by planning."""
        x = float(candidate["x"])
        y = float(candidate["y"])
        self.recent_wander_goals.append((x, y))
        self.wander_goal_publisher.publish(candidate["pose"])

    def set_wandering_active(self, active: bool, force: bool = False) -> None:
        active = bool(active)
        if not force and active == self.wandering_active:
            return
        self.wandering_active = active
        msg = Bool()
        msg.data = active
        self.wandering_active_publisher.publish(msg)
