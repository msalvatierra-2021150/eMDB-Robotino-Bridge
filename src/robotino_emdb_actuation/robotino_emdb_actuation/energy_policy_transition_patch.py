"""
Energy-policy transition logic compatible with the current
RobotinoForagingState.msg interface.

Required RobotinoForagingState fields:
    valid
    visible
    tag_id
    tag_x_map
    tag_y_map
    robot_x_map
    robot_y_map
    robot_yaw_map
    is_energy_bank
    resource_available
    energy_need
    best_energy_tag_id
    best_energy_x_map
    best_energy_y_map
    best_energy_score

Expected publisher:
    self.policy_publisher

Optional immediate exploration control:
    self.set_exploration_enabled(bool)
"""

from robotino_emdb_interfaces.msg import (
    RobotinoForagingState,
    RobotinoSelectedPolicy,
)


POLICY_CONTINUE_EXPLORING = 0
POLICY_INSPECT_VISIBLE_TAG = 1
POLICY_RETURN_TO_BEST_ENERGY_BANK = 2
POLICY_SEARCH_FOR_ENERGY = 3
POLICY_GOAL = 4


class EnergyPolicyTransitionMixin:
    """
    Energy transition:

        energy needed + no bank known
            -> SEARCH_FOR_ENERGY

        energy needed + visible or remembered bank
            -> RETURN_TO_BEST_ENERGY_BANK

    The return policy carries the energy tag map coordinates. The executor
    must approach policy.target_x_map / policy.target_y_map for policy 2.
    """

    def initialize_energy_policy_transition(self):
        self.last_energy_policy_signature = None

    def handle_energy_policy_transition(
        self,
        state: RobotinoForagingState,
    ) -> bool:
        if not state.valid:
            return False

        energy_needed = float(state.energy_need) > 0.0

        if not energy_needed:
            self.last_energy_policy_signature = None
            return False

        visible_energy_bank = bool(
            state.visible
            and state.is_energy_bank
            and state.resource_available
            and int(state.tag_id) >= 0
        )

        remembered_energy_bank = bool(
            int(state.best_energy_tag_id) >= 0
            and float(state.best_energy_score) > 0.0
        )

        if visible_energy_bank:
            # Stop the frontier command immediately instead of waiting for the
            # selected-policy message to loop back through the subscriber.
            if hasattr(self, "set_exploration_enabled"):
                self.set_exploration_enabled(False)

            policy = self.build_visible_energy_bank_policy(state)

        elif remembered_energy_bank:
            if hasattr(self, "set_exploration_enabled"):
                self.set_exploration_enabled(False)

            policy = self.build_return_to_best_energy_bank_policy(state)

        else:
            policy = self.build_search_for_energy_policy(state)

        self.publish_energy_policy_if_changed(policy)
        return True

    def build_search_for_energy_policy(
        self,
        state: RobotinoForagingState,
    ) -> RobotinoSelectedPolicy:
        policy = RobotinoSelectedPolicy()

        policy.header = state.header
        policy.valid = True
        policy.execute_now = True

        policy.policy_id = POLICY_SEARCH_FOR_ENERGY
        policy.policy_name = "search_for_energy"
        policy.use_nav2 = False

        policy.target_tag_id = -1
        policy.target_x_map = 0.0
        policy.target_y_map = 0.0

        # Compatibility fields. Policy 3 does not use these for navigation.

        if hasattr(policy, "target_yaw_map"):
            policy.target_yaw_map = 0.0

        return policy

    def build_visible_energy_bank_policy(
        self,
        state: RobotinoForagingState,
    ) -> RobotinoSelectedPolicy:
        policy = RobotinoSelectedPolicy()

        policy.header = state.header
        policy.valid = True
        policy.execute_now = True

        policy.policy_id = POLICY_RETURN_TO_BEST_ENERGY_BANK
        policy.policy_name = "return_to_visible_energy_bank"
        policy.use_nav2 = True

        policy.target_tag_id = int(state.tag_id)
        policy.target_x_map = float(state.tag_x_map)
        policy.target_y_map = float(state.tag_y_map)

        # These are only fallback/diagnostic fields. Policy 2 navigation uses
        # target_x_map and target_y_map.

        if hasattr(policy, "target_yaw_map"):
            policy.target_yaw_map = 0.0

        return policy

    def build_return_to_best_energy_bank_policy(
        self,
        state: RobotinoForagingState,
    ) -> RobotinoSelectedPolicy:
        policy = RobotinoSelectedPolicy()

        policy.header = state.header
        policy.valid = True
        policy.execute_now = True

        policy.policy_id = POLICY_RETURN_TO_BEST_ENERGY_BANK
        policy.policy_name = "return_to_best_energy_bank"
        policy.use_nav2 = True

        policy.target_tag_id = int(state.best_energy_tag_id)
        policy.target_x_map = float(state.best_energy_x_map)
        policy.target_y_map = float(state.best_energy_y_map)

        # The current state interface does not expose a separate saved robot
        # observation pose for the selected bank. Policy 2 therefore navigates
        # using the remembered tag coordinates above.

        if hasattr(policy, "target_yaw_map"):
            policy.target_yaw_map = 0.0

        return policy

    def publish_energy_policy_if_changed(
        self,
        policy: RobotinoSelectedPolicy,
    ):
        signature = (
            int(policy.policy_id),
            int(policy.target_tag_id),
            round(float(policy.target_x_map), 3),
            round(float(policy.target_y_map), 3),
        )
        if signature == self.last_energy_policy_signature:
            return

        previous_signature = self.last_energy_policy_signature
        self.last_energy_policy_signature = signature

        self.policy_publisher.publish(policy)

        if (
            previous_signature is not None
            and previous_signature[0] == POLICY_SEARCH_FOR_ENERGY
            and policy.policy_id == POLICY_RETURN_TO_BEST_ENERGY_BANK
        ):
            self.get_logger().warn(
                "Energy bank discovered during search. Switching "
                f"policy 3 -> 2 for tag {policy.target_tag_id}."
            )
        else:
            self.get_logger().info(
                f"Published energy policy id={policy.policy_id}, "
                f"name='{policy.policy_name}', "
                f"target_tag={policy.target_tag_id}"
            )