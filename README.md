# Robotino + e-MDB Cognitive Foraging System

**Author / integration:** Michael  Salvatierra

**Credit:** This Robotino integration builds on the e-MDB cognitive architecture and documentation developed in the PILLAR Robots context, with reference components and experiments implemented by **GII**. 

## 1. Overview

This project connects a Festo Robotino running ROS 2 Humble with the e-MDB cognitive architecture. The robot explores an environment, detects semantic AprilTags, stores useful resource information, selects behavior through e-MDB policies, and updates its memory from the result of each action.

The main task is cognitive foraging:

```text
Explore → detect tags → remember resources → select policy → execute navigation → publish outcome → update memory
```

The system is intended for experiments where the robot must discover energy resources, return to them when needed, verify whether they are still present, and eventually reach a final goal marker.

## 2. e-MDB Concepts Used

### Perception

The perception layer converts ROS 2 data into cognitive state information. It reads semantic tag detections, robot state, energy values, known resources, and mapping status.

Main topic:

```bash
/robotino/emdb/tag_detection
```

### Long-Term Memory

The Robotino resource memory stores learned information about tags and resources, including:

- tag ID and semantic type
- estimated map position
- best observation pose
- detection confidence
- number of observations
- reachability evidence
- navigation attempts and failures
- recharge history

Robotino resource memory files:

```bash
~/.robotino_emdb/robotino_resource_memory.yaml
~/.robotino_emdb/robotino_resource_memory.yaml.tmp
```

### Drives

Drives represent internal motivations.

| Drive | Meaning | Typical behavior |
|---|---|---|
| Energy drive | Robot needs to restore energy | return to energy or search for energy |
| Novelty drive | Robot needs new information | explore frontiers or wander |

Energy and novelty can compete. For example, if energy is low, restoring energy should usually have priority over reaching the final goal.

### Goals

The main goals are:

- discover energy resources
- restore energy
- continue exploration
- reach the final goal

### Policies

Policies are executable decisions selected by the e-MDB system and sent to the Robotino policy executor.

| Policy | Purpose |
|---|---|
| `continue_exploring` | Explore with frontier exploration or wandering search |
| `return_to_energy` | Navigate to a remembered energy resource observation/approach pose |
| `verify_energy` | Check whether a remembered energy tag is still visible and reachable |
| `go_to_goal` | Navigate to the final goal marker when conditions are satisfied |

### Policy Outcome

After a policy runs, the executor publishes an outcome. This closes the cognitive loop because the result can update memory and future policy selection.

Main topic:

```bash
/robotino/emdb/policy_outcome
```

Outcomes can include navigation success/failure, tag found/not found, detection confidence, recharge result, and energy before/after recharge.

## 3. Overall Flow

```text
Camera / LiDAR / TF
    ↓
AprilTag detection
    ↓
Robotino tag bridge
    ↓
/robotino/emdb/tag_detection
    ↓
Robotino e-MDB perception
    ↓
Foraging state + Long-Term Memory
    ↓
Drives and goals
    ↓
e-MDB policy selection
    ↓
/robotino/emdb/selected_policy
    ↓
Robotino policy executor
    ↓
Nav2 / frontier exploration / interaction
    ↓
/robotino/emdb/policy_outcome
    ↓
Memory and drive update
```

## 4. Reset the Experiment

### Kill Robotino and e-MDB processes

```bash
pkill -SIGKILL -f 'robotino|emdb|frontier|apriltag'
```

### Restart the ROS 2 daemon

```bash
ros2 daemon stop
ros2 daemon start
```

### Remove Robotino resource memory

```bash
rm -f \
  ~/.robotino_emdb/robotino_resource_memory.yaml \
  ~/.robotino_emdb/robotino_resource_memory.yaml.tmp
```

## 5. Launch the e-MDB System

From the root of the built workspace:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch \
  robotino_emdb_experiments \
  robotino_full_emdb_launch.py
```

Do not place a backslash after `robotino_full_emdb_launch.py` unless more launch arguments follow it.

For real Robotino experiments, keep time consistent:

```bash
use_sim_time:=false
```

## 6. Monitor the Cognitive Loop

In each new terminal:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
```

### Watch the foraging state

```bash
watch -n 2 'ros2 topic echo /robotino/emdb/foraging_state --once'
```

### Watch the selected policy

```bash
watch -n 2 'ros2 topic echo /robotino/emdb/selected_policy --once'
```

### Watch the policy outcome

```bash
watch -n 2 'ros2 topic echo /robotino/emdb/policy_outcome --once'
```

### Watch tag detections

```bash
watch -n 1 'ros2 topic echo /robotino/emdb/tag_detection --once'
```

### Watch frontier exploration enable state

```bash
watch -n 1 'ros2 topic echo /robotino/emdb/frontier_exploration_enable --once'
```

## 7. Expected Policy Sequences

### Normal successful trial

```text
continue_exploring
    ↓
energy tag detected
    ↓
return_to_energy
    ↓
verify_energy
    ↓
recharge succeeds
    ↓
continue_exploring
    ↓
goal tag detected
    ↓
go_to_goal
```

### No known energy resource

```text
low energy
    ↓
continue_exploring
    ↓
discover energy tag
    ↓
return_to_energy
```

### Moved or blocked resource

```text
return_to_energy
    ↓
verify_energy
    ↓
TAG_NOT_FOUND
    ↓
reduce confidence / mark uncertain
    ↓
continue_exploring
    ↓
rediscover resource if visible elsewhere
```

## 8. Recommended Experiment Metrics

Record:

- time to first energy-tag discovery
- policy sequence
- number of selected policies
- navigation successes and failures
- tag verification successes and failures
- detection confidence
- energy before and after recharge
- recharge duration
- reachability confidence changes
- time to adapt after moving or blocking a tag
- time to reach the final goal

These measurements help evaluate whether the robot is only navigating or actually improving its decisions through memory and policy outcomes.

## 9. Acknowledgments

This work gives credit to **GII** for the e-MDB cognitive architecture documentation, reference cognitive nodes/processes, and e-MDB experiment repository. The Robotino integration described here is a custom robotics experiment built on top of those ideas and tools.

Useful references:

- https://docs.pillar-robots.eu/en/latest/
- https://docs.pillar-robots.eu/projects/emdb_experiments_gii/en/latest/index.html
- https://github.com/pillar-robots/emdb_experiments_gii

