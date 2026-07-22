"""Recovery-aware energy-bank eligibility and ranking."""

import math


class EnergyBankRankingMixin:
    """Publish remembered bank facts without exposing unusable recovery targets.

    ``best_energy_tag_id`` and its pose/evidence fields describe the best
    remembered bank, even when that bank is currently empty or cannot finish
    recovery. ``best_energy_score`` is deliberately zero when no remembered
    bank is actionable during recovery. This preserves memory in the publisher
    while allowing the context/decision layer to select ``search_for_energy``.
    """

    def _retry_factor(self, data, now_sec, delay_s):
        """Return a 0..1 verification factor using the newest known evidence."""
        last_information_time = max(
            float(data.get("last_detection_time", 0.0)),
            float(data.get("last_recharge_attempt_time", 0.0)),
        )
        elapsed = (
            float(delay_s)
            if last_information_time <= 0.0
            else max(0.0, now_sec - last_information_time)
        )
        if delay_s <= 0.0:
            return 1.0
        return self.clamp(elapsed / delay_s, 0.0, 1.0)

    @staticmethod
    def _candidate_pose(data):
        """Return the real navigation target and remembered tag pose."""
        tag_x = float(data.get("tag_x_map", math.nan))
        tag_y = float(data.get("tag_y_map", math.nan))
        obs_x = float(data.get("last_seen_robot_x_map", math.nan))
        obs_y = float(data.get("last_seen_robot_y_map", math.nan))
        obs_yaw = float(data.get("last_seen_robot_yaw_map", math.nan))

        if all(math.isfinite(v) for v in (obs_x, obs_y, obs_yaw)):
            target_x = obs_x
            target_y = obs_y
            target_source = "observation"
        elif all(math.isfinite(v) for v in (tag_x, tag_y)):
            target_x = tag_x
            target_y = tag_y
            target_source = "tag"
            obs_x = tag_x
            obs_y = tag_y
            obs_yaw = 0.0
        else:
            return None

        return {
            "tag_x": tag_x if math.isfinite(tag_x) else target_x,
            "tag_y": tag_y if math.isfinite(tag_y) else target_y,
            "obs_x": obs_x,
            "obs_y": obs_y,
            "obs_yaw": obs_yaw,
            "target_x": target_x,
            "target_y": target_y,
            "source": target_source,
        }

    def _verification_score(self, data, supply, now_sec):
        """Return delayed low-priority score material for non-actionable banks."""
        if supply["reason"] == "empty":
            delay_s = self.depleted_retry_delay_s
            factor = self.depleted_verification_score_factor
            mode = "verify_empty"
        else:
            delay_s = self.insufficient_supply_retry_delay_s
            factor = self.insufficient_verification_score_factor
            mode = "verify_insufficient"

        retry_factor = self._retry_factor(data, now_sec, delay_s)
        scoring_resource = (
            float(data.get("resource_capacity", 0.0))
            * factor
            * retry_factor
        )
        return mode, retry_factor, max(0.0, scoring_resource)

    def _write_best_bank(self, state, candidate, actionable_score):
        """Write one remembered candidate to RobotinoForagingState."""
        if candidate is None:
            state.best_energy_tag_id = -1
            state.best_energy_x_map = 0.0
            state.best_energy_y_map = 0.0
            state.best_energy_score = 0.0
            state.best_energy_last_seen_robot_x_map = 0.0
            state.best_energy_last_seen_robot_y_map = 0.0
            state.best_energy_last_seen_robot_yaw_map = 0.0

            optional_defaults = {
                "best_energy_foraging_score": 0.0,
                "best_energy_presence_confidence": 0.0,
                "best_energy_reachability_confidence": 0.0,
                "best_energy_recharge_reliability": 0.0,
                "best_energy_worthiness": 0.0,
                "best_energy_deliverable_energy": 0.0,
                "best_energy_required_for_recovery": float(
                    self.energy_required_for_recovery()
                ),
                "best_energy_sufficient_for_recovery": False,
            }
            for field_name, value in optional_defaults.items():
                self.set_if_available(state, field_name, value)
            return

        pose = candidate["pose"]
        supply = candidate["supply"]
        state.best_energy_tag_id = int(candidate["tag_id"])
        state.best_energy_x_map = float(pose["tag_x"])
        state.best_energy_y_map = float(pose["tag_y"])
        # Positive only when this published candidate is usable by the current
        # decision state. Zero means "remembered, but search for another bank".
        state.best_energy_score = float(max(0.0, actionable_score))
        state.best_energy_last_seen_robot_x_map = float(pose["obs_x"])
        state.best_energy_last_seen_robot_y_map = float(pose["obs_y"])
        state.best_energy_last_seen_robot_yaw_map = float(pose["obs_yaw"])

        self.set_if_available(
            state,
            "best_energy_foraging_score",
            float(candidate["memory_foraging_score"]),
        )
        self.set_if_available(
            state,
            "best_energy_presence_confidence",
            float(candidate["presence"]),
        )
        self.set_if_available(
            state,
            "best_energy_reachability_confidence",
            float(candidate["reachability"]),
        )
        self.set_if_available(
            state,
            "best_energy_recharge_reliability",
            float(candidate["recharge"]),
        )
        self.set_if_available(
            state,
            "best_energy_worthiness",
            float(candidate["worthiness"]),
        )
        self.set_if_available(
            state,
            "best_energy_deliverable_energy",
            float(supply["deliverable"]),
        )
        self.set_if_available(
            state,
            "best_energy_required_for_recovery",
            float(supply["required"]),
        )
        self.set_if_available(
            state,
            "best_energy_sufficient_for_recovery",
            bool(supply["actionable"]),
        )

    def fill_best_energy_bank(self, state):
        """Publish the best actionable bank, or the best remembered fallback.

        Selection has two layers:

        1. An actionable candidate may become a recovery target and receives a
           positive ``best_energy_score``.
        2. If no actionable candidate exists, the best remembered bank is still
           published with its ID, pose and evidence, but its
           ``best_energy_score`` is zero. This preserves factual memory while
           making ``bank_worthy`` false in the context perception, so eMDB runs
           ``search_for_energy`` rather than returning to the depleted bank.
        """
        robot_x = float(getattr(state, "robot_x_map", 0.0))
        robot_y = float(getattr(state, "robot_y_map", 0.0))
        if not math.isfinite(robot_x):
            robot_x = 0.0
        if not math.isfinite(robot_y):
            robot_y = 0.0

        now_sec = self.get_clock().now().nanoseconds / 1e9
        recovery_incomplete = (
            self.energy_required_for_recovery()
            > self.actionable_supply_epsilon
        )

        best_actionable = None
        best_actionable_score = -1.0
        best_remembered = None
        best_remembered_key = None
        candidate_summaries = []

        for tag_id, raw_data in self.memory.items():
            data = self.apply_current_semantics(tag_id, raw_data)

            if not data.get("is_energy_bank", False):
                candidate_summaries.append(f"{tag_id}:not_energy")
                continue

            remaining = float(data.get("resource_remaining", 0.0))
            if not math.isfinite(remaining) or remaining < 0.0:
                candidate_summaries.append(
                    f"{tag_id}:invalid_remaining({remaining!r})"
                )
                continue

            pose = self._candidate_pose(data)
            if pose is None:
                candidate_summaries.append(f"{tag_id}:invalid_pose")
                continue

            distance = math.hypot(
                pose["target_x"] - robot_x,
                pose["target_y"] - robot_y,
            )
            supply = self.evaluate_bank_supply(data, remaining)
            data["last_supply_reason"] = supply["reason"]
            data["last_supply_required"] = float(supply["required"])
            data["last_supply_deliverable"] = float(supply["deliverable"])

            presence = self.presence_confidence(data)
            reachability = self.reachability_confidence(data)
            recharge = self.recharge_reliability(data)
            worthiness = presence * reachability * recharge

            if not all(
                math.isfinite(value)
                for value in (
                    distance,
                    presence,
                    reachability,
                    recharge,
                    worthiness,
                )
            ):
                candidate_summaries.append(f"{tag_id}:non_finite_score")
                continue

            verify_mode, retry_factor, verification_resource = (
                self._verification_score(data, supply, now_sec)
            )

            # This score is descriptive memory information. It is not used to
            # authorize a recovery return when the bank is non-actionable.
            remembered_resource = (
                float(supply["deliverable"])
                if supply["contains_energy"]
                else verification_resource
            )
            memory_foraging_score = remembered_resource / (1.0 + distance)
            memory_score = memory_foraging_score * worthiness

            candidate = {
                "tag_id": int(tag_id),
                "data": data,
                "pose": pose,
                "distance": distance,
                "supply": supply,
                "presence": presence,
                "reachability": reachability,
                "recharge": recharge,
                "worthiness": worthiness,
                "memory_foraging_score": memory_foraging_score,
                "memory_score": memory_score,
                "retry_factor": retry_factor,
                "verification_mode": verify_mode,
            }

            # Always retain one best remembered bank. The tie-breakers ensure
            # even a recently depleted bank with zero verification score still
            # publishes its factual ID and pose.
            remembered_key = (
                float(memory_score),
                float(worthiness),
                float(remaining),
                -float(distance),
                -int(tag_id),
            )
            if best_remembered_key is None or remembered_key > best_remembered_key:
                best_remembered_key = remembered_key
                best_remembered = candidate

            eligible = bool(supply["actionable"])
            candidate_mode = "available" if eligible else supply["reason"]
            scoring_resource = float(supply["deliverable"])

            if not eligible and not recovery_incomplete:
                # Outside recovery, an expired verification cooldown may still
                # produce a low-priority inspection candidate.
                eligible = verification_resource > 0.0
                scoring_resource = verification_resource
                candidate_mode = verify_mode

            if (
                not eligible
                and recovery_incomplete
                and self.allow_nonactionable_verification_during_recovery
            ):
                eligible = verification_resource > 0.0
                scoring_resource = verification_resource
                candidate_mode = verify_mode

            actionable_foraging_score = scoring_resource / (1.0 + distance)
            actionable_score = actionable_foraging_score * worthiness

            if eligible and actionable_score > best_actionable_score:
                best_actionable = candidate
                best_actionable["memory_foraging_score"] = (
                    actionable_foraging_score
                )
                best_actionable_score = actionable_score

            status = "eligible" if eligible else "remembered_only"
            candidate_summaries.append(
                f"{tag_id}:{status}(mode={candidate_mode},"
                f"source={pose['source']},remaining={remaining:.3f},"
                f"deliverable={supply['deliverable']:.3f},"
                f"required={supply['required']:.3f},"
                f"retry={retry_factor:.3f},distance={distance:.3f},"
                f"worthiness={worthiness:.3f},"
                f"target_score={max(0.0, actionable_score):.4f})"
            )

        if best_actionable is not None:
            published = best_actionable
            published_score = best_actionable_score
            publication_mode = "actionable"
        else:
            published = best_remembered
            published_score = 0.0
            publication_mode = "remembered_only"

        self._write_best_bank(state, published, published_score)

        summary = " | ".join(candidate_summaries) or "memory_empty"
        if summary != self.last_best_candidate_summary:
            self.last_best_candidate_summary = summary
            self.get_logger().info(f"Energy-bank candidates: {summary}")

        published_id = -1 if published is None else int(published["tag_id"])
        log_signature = (published_id, publication_mode)
        if log_signature != self.last_logged_best_energy_tag_id:
            self.last_logged_best_energy_tag_id = log_signature
            if published is None:
                self.get_logger().warn(
                    "No remembered energy bank. "
                    f"Candidates: {summary}"
                )
            elif publication_mode == "actionable":
                self.get_logger().info(
                    "Best actionable energy bank -> tag %d | score=%.4f | "
                    "worthiness=%.3f | deliverable=%.3f | required=%.3f | "
                    "observation_pose=(%.2f, %.2f, %.2f)"
                    % (
                        published_id,
                        published_score,
                        published["worthiness"],
                        published["supply"]["deliverable"],
                        published["supply"]["required"],
                        published["pose"]["obs_x"],
                        published["pose"]["obs_y"],
                        published["pose"]["obs_yaw"],
                    )
                )
            else:
                self.get_logger().info(
                    "Best remembered energy bank is not actionable -> "
                    "tag %d | published score=0.0 | reason=%s | "
                    "remaining=%.3f | deliverable=%.3f | required=%.3f. "
                    "Recovery policy should search for another bank."
                    % (
                        published_id,
                        published["supply"]["reason"],
                        published["supply"]["remaining"],
                        published["supply"]["deliverable"],
                        published["supply"]["required"],
                    )
                )