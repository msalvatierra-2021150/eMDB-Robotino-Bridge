# if tag_id not in memory:
#     first_time_seen = True
#     novelty_reward = 1.0
#     memory[tag_id] = {
#         "tag_x_map": msg.tag_x_map,
#         "tag_y_map": msg.tag_y_map,
#         "first_seen_time": now,
#         "last_seen_time": now,
#         "times_seen": 1,
#         "type": tag_semantics[tag_id]["type"],
#         "energy_value": tag_semantics[tag_id]["energy_value"],
#     }
# else:
#     first_time_seen = False
#     novelty_reward = 0.1
#     memory[tag_id]["last_seen_time"] = now
#     memory[tag_id]["times_seen"] += 1

#     energy = energy - exploration_cost_per_second

# if tag_is_energy_bank and robot_is_close_to_tag:
#     energy = min(1.0, energy + tag_energy_value)

#     novelty_reward = 1.0 if first_time_seen else 0.05

# energy_reward = energy_value if tag_is_energy_bank and energy_low else 0.0

# goal_reward = 1.0 if tag_id == target_tag_id else 0.0

# movement_cost = 0.01

# repeated_observation_penalty = 0.1 if times_seen > 3 else 0.0