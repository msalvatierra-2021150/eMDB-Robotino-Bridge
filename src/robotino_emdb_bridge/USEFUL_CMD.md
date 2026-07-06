April Tag camera perception robotino - eMDB Bridge

ros2 run robotino_emdb_bridge apriltag_tf_to_emdb_bridge --ros-args \
  -p detections_topic:=/detections \
  -p output_topic:=/robotino/emdb/tag_observation \
  -p map_frame:=map \
  -p robot_frame:=base_link \
  -p camera_frame:=camera_optical_frame

To see the topic
ros2 topic echo /robotino/emdb/tag_observation

Run Foragin Memory
ros2 run robotino_emdb_memory foraging_memory

Launch Cognitive nodes
ros2 launch robotino_emdb_experiments robotino_semantic_experiment_launch.py

See Foraging State Values
ros2 topic echo /perception/foraging_state/value
