std_msgs/Header header

bool valid

float32 novelty_drive
float32 energy_drive
float32 goal_drive

int32 dominant_drive_id
string dominant_drive

int32 suggested_policy_id
string suggested_policy

string suggested_goal

int32 target_tag_id
float32 target_x_map
float32 target_y_map

float32 priority_confidence

float32 exploration_priority
float32 inspect_priority
float32 return_priority
float32 search_energy_priority

bool should_interrupt_exploration
bool should_send_nav_goal