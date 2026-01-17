# [2026-01-16] UR5e Gripper Slipping Fix & Environment Robustness

## 1. Problem Description
- **Issue**: The UR5e robot with 2F85 gripper was experiencing significant object slipping and detachment in the `pick_box` environment, making data collection impossible.
- **Context**: The `pick_box_scene_copy.xml` scene included complex cloth simulation elements (`mujoco3_cloth.xml` and constraints) alongside the main task.

## 2. Root Cause Analysis
- **Diagnosis**: The cloth simulation introduced excessive computational overhead and physics instability (confirmed by "rate limiter late" warnings and direct observation). This instability interfered with the contact dynamics between the gripper pads and the box.
- **Verification**: Removing the cloth elements from the scene immediately resolved the slipping issue, enabling stable grasping.

## 3. Implementation Changes

### Environment Code (`pick_box_env.py`)
- **Objective**: Improve robustness so the environment supports dynamic toggling of XML assets without crashing.
- **Modifications**:
  - **`step()` Function**: Added `try-except` blocks and a `cloth_exists` flag. If cloth bodies are missing, the cloth attachment logic is safely skipped.
  - **`reset()` Function**: Added `try-except` blocks around SO101 initialization. If the SO101 robot is missing from the XML, the initialization gracefully handles the absence using default values.

### Scene XML (`pick_box_scene_copy.xml`)
- Commented out the `mujoco3_cloth.xml` include.
- Commented out the `so101_new_calib.xml` (SO101 robot) include.
- Commented out all associated `weld` equality constraints for cloth and SO101.

## 4. Related Work: SO101 Environment (`so101_pick_box`)
*Recent experiments performed by user:*
- **Physics Settings**:
  - Changed `timestep` from `0.02` to `0.002`.
  - Changed `solver` from `CG` to `PGS`.
- **Collision Geometry (`so101_new_calib.xml`)**:
  - Commented out explicit collision box pads (`so101_fixed_pad`, `so101_moving_pad`).
  - Enabled mesh-based collision (`so101_collision`).
- **Scene Objects**:
  - Modified table setup and removed explicit friction overrides for the Box.

## 5. Result
- **UR5e**: User confirmed **"在註解掉布料後,夾取就完全正常了"** (Grasping is completely normal after commenting out cloth).
- **Status**: The `pick_box` environment is now stable and ready for data collection. The `pick_box_env.py` is robust enough to handle future configuration changes.
