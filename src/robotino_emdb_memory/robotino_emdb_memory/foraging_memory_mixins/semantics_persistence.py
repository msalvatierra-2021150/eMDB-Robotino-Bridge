"""Semantic configuration and persistent factual-memory support."""

from pathlib import Path

import yaml

class SemanticsPersistenceMixin:
    """Mixin extracted from RobotinoForagingMemory."""

    def load_tag_semantics(self, semantics_file):
        path = Path(semantics_file)

        if not path.exists():
            self.get_logger().warn(
                f"Semantics file not found: {semantics_file}. Using defaults."
            )
            return {
                0: {
                    "type": "landmark",
                    "is_energy_bank": False,
                    "capacity": 0.0,
                    "collection_rate": 0.0,
                    "regen_rate": 0.0,
                },
                1: {
                    "type": "low_energy_bank",
                    "is_energy_bank": True,
                    "capacity": 0.30,
                    "collection_rate": 0.08,
                    "regen_rate": 0.0,
                },
                2: {
                    "type": "medium_energy_bank",
                    "is_energy_bank": True,
                    "capacity": 0.50,
                    "collection_rate": 0.12,
                    "regen_rate": 0.0,
                },
                3: {
                    "type": "checkpoint",
                    "is_energy_bank": False,
                    "capacity": 0.0,
                    "collection_rate": 0.0,
                    "regen_rate": 0.0,
                },
                4: {
                    "type": "high_energy_bank",
                    "is_energy_bank": True,
                    "capacity": 0.80,
                    "collection_rate": 0.18,
                    "regen_rate": 0.0,
                },
                5: {
                    "type": "goal_marker",
                    "is_energy_bank": False,
                    "capacity": 0.0,
                    "collection_rate": 0.0,
                    "regen_rate": 0.0,
                },
            }

        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}

        semantics = {}
        for raw_id, tag_data in data.get("tags", {}).items():
            tag_id = int(raw_id)
            semantics[tag_id] = {
                "type": str(tag_data.get("type", "unknown")),
                "is_energy_bank": bool(
                    tag_data.get("is_energy_bank", False)
                ),
                "capacity": float(tag_data.get("capacity", 0.0)),
                "collection_rate": float(
                    tag_data.get("collection_rate", 0.0)
                ),
                "regen_rate": float(tag_data.get("regen_rate", 0.0)),
            }

        self.get_logger().info(
            f"Loaded tag semantics from: {semantics_file}"
        )
        return semantics

    def create_resource_truth(self):
        """Create private physical resource state from semantic YAML.

        This state owns regeneration. It is intentionally separate from
        ``self.memory`` so Robotino cannot infer availability from regen_rate.
        """
        resources = {}
        for tag_id, semantics in self.tag_semantics.items():
            if not bool(semantics.get("is_energy_bank", False)):
                continue

            capacity = max(0.0, float(semantics.get("capacity", 0.0)))
            resources[int(tag_id)] = {
                "capacity": capacity,
                "remaining": capacity,
                "collection_rate": max(
                    0.0,
                    float(semantics.get("collection_rate", 0.0)),
                ),
                "regen_rate": max(
                    0.0,
                    float(semantics.get("regen_rate", 0.0)),
                ),
                "last_update_time": 0.0,
            }
        return resources

    def apply_current_semantics(self, tag_id, data):
        """Migrate a remembered record to the currently loaded semantics.

        Earlier runs may have stored a tag as ``unknown`` or as a non-energy
        tag when the semantics path or tag IDs were wrong.  Without this
        migration, correcting the YAML would never make that persistent record
        eligible for best-bank selection.
        """
        semantics = self.tag_semantics.get(int(tag_id))
        if semantics is None:
            if int(tag_id) not in self.warned_unknown_tag_ids:
                self.warned_unknown_tag_ids.add(int(tag_id))
                self.get_logger().warn(
                    f"Tag {int(tag_id)} is not present in "
                    f"{self.semantics_file}; it cannot be an energy bank."
                )
            return self.ensure_memory_schema(data)

        previous_type = str(data.get("tag_type", "unknown"))
        previous_is_bank = bool(data.get("is_energy_bank", False))

        data["tag_type"] = str(semantics["type"])
        data["is_energy_bank"] = bool(semantics["is_energy_bank"])
        data["resource_capacity"] = float(semantics["capacity"])
        data["collection_rate"] = float(semantics["collection_rate"])
        # Remove legacy privileged knowledge from persistent learned memory.
        data.pop("regen_rate", None)
        data.setdefault("resource_remaining", float(semantics["capacity"]))

        # Clamp old persisted values to the current configured capacity.
        capacity = max(0.0, float(data["resource_capacity"]))
        data["resource_remaining"] = self.clamp(
            data.get("resource_remaining", capacity),
            0.0,
            capacity,
        )

        if (
            previous_type != data["tag_type"]
            or previous_is_bank != data["is_energy_bank"]
        ):
            self.get_logger().warn(
                f"Migrated tag {int(tag_id)} semantics: "
                f"type {previous_type!r} -> {data['tag_type']!r}, "
                f"is_energy_bank {previous_is_bank} -> "
                f"{data['is_energy_bank']}"
            )

        return self.ensure_memory_schema(data)

    def load_resource_memory(self):
        if not self.persist_memory or not self.memory_file.exists():
            return

        try:
            with self.memory_file.open("r", encoding="utf-8") as file:
                payload = yaml.safe_load(file) or {}

            raw_tags = payload.get("tags", {})
            for raw_id, data in raw_tags.items():
                tag_id = int(raw_id)
                record = self.apply_current_semantics(
                    tag_id, dict(data)
                )

                # ROS simulation time may restart between runs. Keep the facts
                # and evidence, but restart transient time references.
                record["last_seen_time"] = 0.0
                record["last_detection_time"] = 0.0
                record.pop("last_resource_update_time", None)
                record.pop("last_collection_time", None)
                record.pop("regen_rate", None)
                self.memory[tag_id] = record

            self.get_logger().info(
                f"Loaded {len(self.memory)} remembered tags"
            )
        except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
            self.get_logger().error(
                f"Could not load Robotino resource memory: {error}"
            )

    def restore_resource_truth_from_memory(self):
        """Resume hidden resource amounts from the last observed values."""
        for tag_id, data in self.memory.items():
            resource = self.resource_truth.get(int(tag_id))
            if resource is None:
                continue
            resource["remaining"] = self.clamp(
                data.get("resource_remaining", resource["capacity"]),
                0.0,
                resource["capacity"],
            )
            resource["last_update_time"] = 0.0
            resource.pop("last_collection_time", None)

    def save_resource_memory(self):
        if not self.persist_memory:
            return

        try:
            self.memory_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "tags": {
                    int(tag_id): data
                    for tag_id, data in sorted(self.memory.items())
                }
            }
            temporary = self.memory_file.with_suffix(
                self.memory_file.suffix + ".tmp"
            )
            with temporary.open("w", encoding="utf-8") as file:
                yaml.safe_dump(payload, file, sort_keys=True)
            temporary.replace(self.memory_file)
        except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
            self.get_logger().error(
                f"Could not save Robotino resource memory: {error}"
            )
