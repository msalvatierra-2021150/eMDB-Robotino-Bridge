# Robotino eMDB System Guide

## Overview

This project integrates the **eMDB cognitive architecture** with a Festo Robotino running ROS 2 Humble.

The system allows the robot to:

* Explore an unknown or partially known environment.
* Detect semantic resources using AprilTags.
* Store information about previously observed resources.
* Monitor internal needs such as energy and novelty.
* Select policies based on its current state and active goals.
* Navigate toward remembered resources.
* Verify whether a resource is still present.
* Recharge when an energy resource is available.
* Adapt its behavior based on policy outcomes.
* Navigate toward the final goal when the required conditions are satisfied.

The complete cognitive loop is:

```text
Environment
    ↓
Robot sensors and AprilTag detector
    ↓
Robotino eMDB perception
    ↓
Foraging state and Long-Term Memory
    ↓
Drives and goals
    ↓
Policy selection
    ↓
Policy execution
    ↓
Navigation and interaction
    ↓
Policy outcome
    ↓
Memory and learning update
    ↺
```

---

# 1. Main Components

## 1.1 Perception

The perception layer converts information from the robot and environment into data that the eMDB can use.

Examples of perceived information include:

* Current robot energy.
* Energy need.
* Whether an AprilTag is visible.
* Tag ID and semantic type.
* Tag confidence.
* Distance and bearing to the tag.
* Tag position in the map.
* Robot position when the tag was observed.
* Known resource locations.
* Resource availability.
* Exploration status.
* Whether the environment has been fully mapped.
* Whether the final goal has been detected.

The main semantic detection topic is:

```text
/robotino/emdb/tag_detection
```

The eMDB perception node reads this information and creates the current cognitive state used by the commander, drives, goals, and policies.

---

## 1.2 Foraging State

The foraging state summarizes the robot's current understanding of the task.

Topic:

```text
/robotino/emdb/foraging_state
```

It may contain information such as:

* Current robot energy.
* Current energy need.
* Visible tag information.
* Best known energy resource.
* Position of known resources.
* Whether a resource is expected to be available.
* Resource capacity and remaining energy.
* Resource reachability.
* Novelty reward.
* Energy reward.
* Goal reward.
* Total reward.
* Whether the final goal has been satisfied.

The foraging state is one of the most useful topics for understanding what the cognitive system currently believes.

Monitor it with:

```bash
watch -n 2 'ros2 topic echo /robotino/emdb/foraging_state --once'
```

---

## 1.3 Long-Term Memory

The eMDB Long-Term Memory, or LTM, stores experience accumulated during robot operation.

In this Robotino integration, memory can contain information such as:

* Previously observed tag IDs.
* Semantic type of each tag.
* Estimated map position.
* Robot pose from which the tag was successfully observed.
* Best observation confidence.
* Number of observations.
* Navigation attempts.
* Navigation successes and failures.
* Resource availability.
* Reachability evidence.
* Recharge history.
* Whether a resource may be temporarily unreachable.
* Whether the location should be verified again.

The saved Robotino resource memory is stored in:

```text
~/.robotino_emdb/robotino_resource_memory.yaml
```

A temporary file may also exist:

```text
~/.robotino_emdb/robotino_resource_memory.yaml.tmp
```

The memory is used so that the robot does not need to rediscover every resource from zero during every decision cycle.

---

## 1.4 P-Nodes

The eMDB can use structures called **P-Nodes** to associate:

* A perceived world state.
* An active goal.
* A policy.
* The outcome or reward obtained after executing that policy.

For example, the system may learn that:

```text
Low energy
+ known reachable energy resource
+ return_to_energy policy
→ successful recharge
```

The commander logs may show messages similar to:

```text
Added point in pnode pnode_robotino_world__discover_energy_resource_goal__return_to_energy
```

This means that an experience point was added to the P-Node associated with that world, goal, and policy relationship.

As more experiences are collected, the eMDB can improve the relationship between the current state and the policy that is most appropriate.

---

# 2. Drives

Drives represent internal pressures or motivations that influence robot behavior.

The main drives used in this project are the **energy drive** and the **novelty drive**.

## 2.1 Energy Drive

The energy drive represents the robot's need to maintain or restore its energy.

Its value normally increases as the robot's energy decreases.

Conceptually:

```text
High robot energy → low energy drive
Low robot energy  → high energy drive
```

When the energy drive becomes important, the robot should prioritize finding or returning to an energy resource.

The energy drive can influence behaviors such as:

* Searching for an energy tag.
* Returning to a known energy resource.
* Verifying that a remembered energy resource is still present.
* Recharging.
* Resuming exploration after the energy need has been satisfied.

Typical logs may include:

```text
[energy_drive]: RESETTING REWARD
```

A reward reset usually indicates that the drive evaluation or its associated reward has changed because of a new state or completed action.

---

## 2.2 Novelty Drive

The novelty drive motivates the robot to acquire new information.

It is especially important when:

* The map is incomplete.
* Unknown areas remain.
* No useful energy resource is known.
* New semantic tags may still exist.
* The robot has not recently discovered anything new.

The novelty drive supports behaviors such as:

* Frontier exploration.
* Wandering after frontier-based mapping is complete.
* Visiting unexplored regions.
* Searching for semantic resources.
* Continuing exploration even when energy is currently sufficient.

This means exploration does not have to occur only when the robot needs energy.

The robot can explore because:

```text
Novelty need is active
```

or because:

```text
The robot needs to discover an energy resource
```

These are related but different motivations.

---

## 2.3 Interaction Between Drives

The selected behavior depends on which need is currently more important.

Example:

```text
Energy is sufficient
+ novelty is high
→ continue exploring
```

```text
Energy is low
+ a known energy resource exists
→ return to energy
```

```text
Energy is low
+ no energy resource is known
→ explore to discover a resource
```

```text
Energy has been restored
+ final goal is known
+ task conditions are satisfied
→ go to goal
```

The exact decision depends on the goal configuration, policy evaluation, learned experience, and the current foraging state.

---

# 3. Goals

Goals define what the cognitive system is currently trying to accomplish.

In this project, the important goals include discovering energy resources, restoring energy, exploring the environment, and reaching the final target.

## 3.1 Discover Energy Resource Goal

This goal becomes important when the robot needs energy but does not know a usable energy resource.

Possible policy:

```text
continue_exploring
```

Expected result:

* The robot explores.
* An energy tag is detected.
* The resource is added to memory.
* The robot can then transition to returning to that resource.

---

## 3.2 Restore Energy Goal

This goal becomes active when the robot's energy is below the desired level.

Possible policies include:

```text
return_to_energy
verify_energy
```

Expected result:

* The robot returns to a remembered observation location.
* It attempts to observe the energy tag again.
* It approaches or interacts with the energy resource.
* The robot's energy increases.
* The energy drive decreases.

---

## 3.3 Exploration or Novelty Goal

This goal encourages continued acquisition of information.

Possible policy:

```text
continue_exploring
```

Exploration can happen in two ways:

1. **Frontier exploration**

   Used while unexplored frontiers remain in the occupancy map.

2. **Wandering search**

   Used after the map is considered complete or when no valid frontier is available.

---

## 3.4 Final Goal

The final goal represents completion of the experiment or mission.

Possible policy:

```text
go_to_goal
```

The robot should normally treat this as a lower-priority objective than maintaining sufficient energy.

A reasonable priority structure is:

```text
1. Prevent critical energy depletion
2. Discover or return to energy resources
3. Continue exploration and acquire knowledge
4. Navigate to the final goal
```

---

# 4. Policies

Policies are actions or behavioral strategies selected by the eMDB commander.

The policy selected by the cognitive system is published on:

```text
/robotino/emdb/selected_policy
```

Monitor it with:

```bash
watch -n 2 'ros2 topic echo /robotino/emdb/selected_policy --once'
```

The policy names below are specific to this Robotino integration.

---

## 4.1 `continue_exploring`

### Purpose

Continue autonomous exploration and search for new semantic information.

### Typical conditions

* The robot has sufficient energy.
* The novelty drive is active.
* No usable energy resource is known.
* The robot needs to discover an energy resource.
* The final goal is not yet ready to be completed.

### Execution

The policy executor enables frontier exploration:

```text
/robotino/emdb/frontier_exploration_enable = true
```

If frontier-based exploration has completed, the system may use wandering behavior instead.

### Expected outcomes

* New area explored.
* New AprilTag detected.
* Energy resource discovered.
* Goal marker discovered.
* Mapping completed.
* Exploration continues without a terminal result.

---

## 4.2 `return_to_energy`

### Purpose

Navigate back toward a previously observed energy resource.

### Typical conditions

* The energy drive is active.
* A known energy resource exists.
* The resource has a remembered observation pose.
* The resource is considered reachable or worth retrying.

### Execution

The robot should normally navigate to a safe observation or approach pose instead of navigating directly to the tag coordinates.

This is important because the tag may be attached to a wall.

The remembered data can include:

```text
Tag map position
Robot pose when the tag was observed
Best observation confidence
Approach standoff distance
```

The navigation target is calculated to allow the robot to see or approach the resource without colliding with the wall.

### Expected outcomes

* Navigation succeeds.
* The tag is visible again.
* The robot is within interaction distance.
* The resource can be verified or used.

---

## 4.3 `verify_energy`

### Purpose

Confirm whether a remembered energy resource still exists and is available.

### Typical conditions

* The robot reached the remembered resource area.
* The resource was not immediately visible.
* The environment may have changed.
* A previous navigation or recharge attempt failed.
* Memory confidence is uncertain.

### Execution

The robot may:

* Rotate or adjust its pose.
* Search around the remembered observation point.
* Check for the expected tag.
* Compare the observed resource with the remembered resource.
* Update presence and reachability evidence.

### Expected outcomes

```text
TAG_FOUND
```

or:

```text
TAG_NOT_FOUND
```

If the tag is found, the robot may continue with resource interaction.

If the tag is not found, memory confidence can be reduced and the robot may resume exploration.

---

## 4.4 `go_to_goal`

### Purpose

Navigate toward the final goal marker.

### Typical conditions

* The goal tag is known.
* The robot has sufficient energy.
* Higher-priority energy needs are satisfied.
* Required exploration or mission conditions have been completed.

### Execution

Like energy resources, a goal marker may be mounted on a wall.

The executor should therefore use:

* A remembered observation pose.
* A safe standoff distance.
* A navigable approach pose.
* Costmap and line-of-sight validation.

### Expected outcomes

* Robot reaches the final goal area.
* Goal tag is confirmed.
* Goal reward is generated.
* Mission is marked as complete.

---

# 5. Policy Outcome

After executing a policy, the executor publishes the result on:

```text
/robotino/emdb/policy_outcome
```

Monitor it with:

```bash
watch -n 2 'ros2 topic echo /robotino/emdb/policy_outcome --once'
```

The policy outcome may contain:

* Policy ID.
* Navigation result.
* Tag verification result.
* Detection confidence.
* Observed tag pose.
* Whether recharge was attempted.
* Whether recharge succeeded.
* Energy before recharge.
* Energy after recharge.
* Recharge duration.
* Failure or cancellation information.

Important result values may include:

```text
Navigation:
  SUCCEEDED
  FAILED
  CANCELED
```

```text
Tag verification:
  TAG_FOUND
  TAG_NOT_FOUND
```

The outcome closes the cognitive loop by providing feedback to memory and learning.

---

# 6. Overall Execution Flow

## 6.1 Exploration Flow

```text
Robot starts
    ↓
Foraging perception reads current state
    ↓
Novelty or discovery goal becomes active
    ↓
eMDB selects continue_exploring
    ↓
Policy bridge publishes selected policy
    ↓
Policy executor enables frontier exploration
    ↓
Robot moves toward frontiers
    ↓
AprilTag detector observes semantic tag
    ↓
Tag detection is transformed into map coordinates
    ↓
Foraging state and LTM are updated
```

---

## 6.2 Energy Recovery Flow

```text
Robot energy decreases
    ↓
Energy drive increases
    ↓
Restore-energy goal becomes important
    ↓
Memory checks for known energy resources
    ↓
Known energy resource exists?
```

### Known resource exists

```text
Yes
 ↓
Select return_to_energy
 ↓
Disable exploration
 ↓
Navigate to remembered observation or approach pose
 ↓
Observe expected tag
 ↓
Verify resource
 ↓
Recharge
 ↓
Publish policy outcome
 ↓
Update memory and energy drive
 ↓
Resume exploration or continue toward goal
```

### No known resource exists

```text
No
 ↓
Select continue_exploring
 ↓
Search for an energy resource
 ↓
Detect energy tag
 ↓
Store resource in memory
 ↓
Select return_to_energy
```

---

## 6.3 Nonstationary Resource Flow

The environment may change after a resource has been learned.

For example:

* A tag is moved.
* A resource is removed.
* An obstacle blocks the previous approach.
* A resource becomes temporarily unreachable.

The expected behavior is:

```text
Navigate to remembered observation area
    ↓
Expected tag is not visible
    ↓
Verify the area
    ↓
Tag still not found
    ↓
Publish TAG_NOT_FOUND
    ↓
Reduce presence or reachability confidence
    ↓
Mark resource as uncertain or temporarily unreachable
    ↓
Resume exploration
    ↓
Rediscover the resource if it appears elsewhere
```

This experiment is useful for measuring adaptation and memory updating.

---

## 6.4 Goal Completion Flow

```text
Goal tag has been discovered
    ↓
Robot checks current energy
    ↓
Energy sufficient?
```

```text
No
 ↓
Restore energy first
```

```text
Yes
 ↓
Select go_to_goal
 ↓
Navigate to safe goal observation pose
 ↓
Confirm goal tag
 ↓
Publish successful outcome
 ↓
Goal reward is generated
 ↓
Mission completes
```

---

# 7. ROS 2 Data Flow

```mermaid
flowchart TD
    A[Camera and Robot Sensors] --> B[AprilTag Detection]
    B --> C[AprilTag TF to eMDB Bridge]
    C --> D[/robotino/emdb/tag_detection]

    D --> E[Robotino Foraging Perception]
    E --> F[Foraging State]
    E --> G[eMDB Long-Term Memory]

    F --> H[Drives]
    G --> H

    H --> I[Goals]
    I --> J[eMDB Commander and Policy Selection]

    J --> K[/robotino/emdb/selected_policy]
    K --> L[Policy Execution Bridge]
    L --> M[Robotino Policy Executor]

    M --> N[Nav2 and Frontier Exploration]
    N --> O[Robot Motion and Environment Interaction]

    O --> P[/robotino/emdb/policy_outcome]
    P --> G
    P --> H
```

A simplified version is:

```text
Tag detection
    ↓
Perception
    ↓
Memory and foraging state
    ↓
Drives
    ↓
Goals
    ↓
Policy selection
    ↓
Policy executor
    ↓
Nav2 or exploration
    ↓
Policy outcome
    ↓
Memory update
```

---

# 8. Important ROS 2 Topics

| Topic                                        | Purpose                                       |
| -------------------------------------------- | --------------------------------------------- |
| `/detections`                                | Raw AprilTag detections                       |
| `/robotino/emdb/tag_detection`               | Semantic tag observation sent to the eMDB     |
| `/robotino/emdb/foraging_state`              | Current cognitive and foraging state          |
| `/robotino/emdb/selected_policy`             | Policy selected for execution                 |
| `/robotino/emdb/policy_outcome`              | Result of the executed policy                 |
| `/robotino/emdb/frontier_exploration_enable` | Enables or disables frontier exploration      |
| `/frontier_exploration/mapping_complete`     | Indicates that frontier mapping has completed |
| `/scan`                                      | LiDAR data                                    |
| `/map`                                       | Occupancy map                                 |
| `/tf`                                        | Coordinate transformations                    |
| `/camera/image_rect`                         | Rectified camera image                        |
| `/camera/camera_info`                        | Camera calibration information                |

---

# 9. Stop the Complete Robotino eMDB System

Kill Robotino, eMDB, frontier exploration, and AprilTag processes:

```bash
pkill -SIGKILL -f 'robotino|emdb|frontier|apriltag'
```

Restart the ROS 2 daemon:

```bash
ros2 daemon stop
ros2 daemon start
```

Verify that no matching processes remain:

```bash
ps aux | grep -E 'robotino|emdb|frontier|apriltag' | grep -v grep
```

No output means that the matching processes have been terminated.

> `SIGKILL` should mainly be used when the nodes do not stop normally. When possible, stop the launch process first with `Ctrl+C`.

---

# 10. Remove the Robotino Resource Memory

Delete the saved memory and its temporary file:

```bash
rm -f \
  ~/.robotino_emdb/robotino_resource_memory.yaml \
  ~/.robotino_emdb/robotino_resource_memory.yaml.tmp
```

Verify the directory:

```bash
ls -la ~/.robotino_emdb/
```

Deleting these files resets the custom Robotino resource memory.

The robot will need to rediscover:

* Energy resources.
* Goal tags.
* Resource positions.
* Observation poses.
* Reachability information.
* Recharge history.

This may not remove every internal eMDB LTM artifact if the eMDB core is configured to save additional memory in another location. It specifically removes the Robotino resource-memory files shown above.

---

# 11. Launch the eMDB System

Open a terminal in the root of the built eMDB workspace.

Source ROS 2 Humble:

```bash
source /opt/ros/humble/setup.bash
```

Source the current workspace:

```bash
source install/setup.bash
```

Launch the complete Robotino eMDB system:

```bash
ros2 launch \
  robotino_emdb_experiments \
  robotino_full_emdb_launch.py
```

Do not add a backslash after the launch filename unless more launch arguments follow it.

Example with a launch argument:

```bash
ros2 launch \
  robotino_emdb_experiments \
  robotino_full_emdb_launch.py \
  use_sim_time:=false
```

For the real Robotino system, use real time consistently:

```text
use_sim_time:=false
```

All related nodes should use the same time source.

---

# 12. Monitor the Cognitive System

In every new terminal, source ROS 2 and the workspace first:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
```

## Watch the foraging state

```bash
watch -n 2 'ros2 topic echo /robotino/emdb/foraging_state --once'
```

## Watch the selected policy

```bash
watch -n 2 'ros2 topic echo /robotino/emdb/selected_policy --once'
```

## Watch the policy outcome

```bash
watch -n 2 'ros2 topic echo /robotino/emdb/policy_outcome --once'
```

## Watch tag detections

```bash
watch -n 1 'ros2 topic echo /robotino/emdb/tag_detection --once'
```

## Watch whether exploration is enabled

```bash
watch -n 1 'ros2 topic echo /robotino/emdb/frontier_exploration_enable --once'
```

## Watch mapping completion

```bash
watch -n 2 'ros2 topic echo /frontier_exploration/mapping_complete --once'
```

---

# 13. Recommended Terminal Layout

| Terminal   | Purpose                                            |
| ---------- | -------------------------------------------------- |
| Terminal 1 | Launch `robotino_full_emdb_launch.py`              |
| Terminal 2 | Monitor `/robotino/emdb/foraging_state`            |
| Terminal 3 | Monitor `/robotino/emdb/selected_policy`           |
| Terminal 4 | Monitor `/robotino/emdb/policy_outcome`            |
| Terminal 5 | Monitor tag detections or exploration enable state |
| Terminal 6 | Inspect Nav2, TF, and system diagnostics           |

For multiple monitoring panes, `tmux` can make the experiment easier to observe.

---

# 14. Complete Reset and Launch Sequence

```bash
# Stop related processes
pkill -SIGKILL -f 'robotino|emdb|frontier|apriltag'

# Restart ROS 2 discovery
ros2 daemon stop
ros2 daemon start

# Remove saved Robotino resource memory
rm -f \
  ~/.robotino_emdb/robotino_resource_memory.yaml \
  ~/.robotino_emdb/robotino_resource_memory.yaml.tmp

# Source ROS 2
source /opt/ros/humble/setup.bash

# Source the eMDB workspace
source install/setup.bash

# Launch the complete system
ros2 launch \
  robotino_emdb_experiments \
  robotino_full_emdb_launch.py
```

---

# 15. Experiment Startup Checklist

Before starting an experiment, verify:

```text
[ ] Old Robotino and eMDB processes are stopped
[ ] ROS 2 daemon has been restarted
[ ] Resource memory was deleted if a clean trial is required
[ ] ROS 2 Humble has been sourced
[ ] The correct workspace has been sourced
[ ] use_sim_time is false on all real-robot nodes
[ ] Camera images are publishing
[ ] LiDAR scans are publishing
[ ] TF tree is available
[ ] Nav2 is active
[ ] AprilTag detections are publishing
[ ] Foraging state is publishing
[ ] Selected policy is publishing
[ ] Policy outcome is publishing
[ ] Frontier exploration can be enabled and disabled
```

Useful checks:

```bash
ros2 topic list
```

```bash
ros2 node list
```

```bash
ros2 topic hz /scan
```

```bash
ros2 topic hz /camera/image_rect
```

```bash
ros2 topic info /robotino/emdb/tag_detection -v
```

```bash
ros2 topic info /robotino/emdb/selected_policy -v
```

```bash
ros2 topic info /robotino/emdb/policy_outcome -v
```

---

# 16. Expected Policy Sequence During a Normal Trial

A typical successful experiment may produce a sequence similar to:

```text
continue_exploring
    ↓
Energy tag detected
    ↓
return_to_energy
    ↓
verify_energy
    ↓
Recharge succeeds
    ↓
continue_exploring
    ↓
Goal tag detected
    ↓
go_to_goal
    ↓
Mission complete
```

When no energy resource is known:

```text
Low energy
    ↓
continue_exploring
    ↓
Discover energy resource
    ↓
return_to_energy
```

When a remembered resource has moved:

```text
return_to_energy
    ↓
verify_energy
    ↓
TAG_NOT_FOUND
    ↓
Memory confidence decreases
    ↓
continue_exploring
    ↓
Resource rediscovered at new location
```

---

# 17. What to Record During Experiments

For research and paper evaluation, record:

* Initial robot energy.
* Time required to discover the first resource.
* Number of explored frontiers.
* Number of tags detected.
* Detection confidence.
* Resource position error.
* Number of selected policies.
* Policy transition sequence.
* Navigation attempts.
* Navigation successes and failures.
* Resource verification successes and failures.
* Recharge attempts and successes.
* Energy before and after recharge.
* Time required to restore energy.
* Number of times a missing resource is retried.
* Time required to adapt after moving a resource.
* Number of unnecessary policy switches.
* Time required to reach the final goal.
* Final accumulated reward.
* Changes in P-Node or LTM data.

These measurements make it possible to evaluate not only whether the robot completed the task, but also how the cognitive architecture changed its behavior based on experience.
