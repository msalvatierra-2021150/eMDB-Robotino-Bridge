April Tag camera perception robotino - eMDB Bridge

ros2 run robotino_emdb_bridge apriltag_tf_to_emdb_bridge --ros-args   
-p detections_topic:=/detections   
-p output_topic:=/robotino/emdb/tag_detection   
-p map_frame:=map   
-p robot_frame:=base_link   
-p camera_frame:=camera_optical_frame

To see the topic

ros2 topic echo /robotino/emdb/tag_detection