std_msgs/Header header

bool valid
bool visible

int32 tag_id
string tag_type

float32 confidence
float32 distance
float32 bearing

float32 tag_x_map
float32 tag_y_map
float32 tag_yaw_map

float32 robot_x_map
float32 robot_y_map
float32 robot_yaw_map

bool first_time_seen
bool known_tag
uint32 times_seen
float32 time_since_last_seen

bool is_energy_bank
float32 resource_capacity
float32 resource_remaining
float32 resource_value
bool resource_available

float32 robot_energy
float32 energy_need

float32 novelty_reward
float32 energy_reward
float32 goal_reward
float32 total_reward

int32 best_energy_tag_id
float32 best_energy_x_map
float32 best_energy_y_map
float32 best_energy_score

bool goal_satisfied