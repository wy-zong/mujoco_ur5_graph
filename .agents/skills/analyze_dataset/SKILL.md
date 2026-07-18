---
name: Analyze LeRobot Dataset
description: Analyze a LeRobot parquet dataset to calculate total episodes, frames, and moving vs stationary durations based on robot joint states.
---

# Analyze LeRobot Dataset

This skill analyzes a LeRobot `.parquet` dataset directory (`outputs/...`) by reading `meta/info.json` and the `.parquet` data files to calculate episode lengths and the duration the robot spends moving vs stationary.

## Usage

1. **Locate the target dataset directory**, e.g., `C:\Users\ccu\mujoco_ur5_graph\outputs\dataset\...`
2. Run the `analyze.py` script provided in this skill's `scripts` directory on the dataset directory.

// turbo
```bash
python .agents/skills/analyze_dataset/scripts/analyze.py "<DATASET_DIRECTORY_PATH>"
```

## How it works
The script:
1. Reads `meta/info.json` to get the `fps`.
2. Iterates over all `.parquet` files in the `data/` subdirectory.
3. Loads the `observation.state` column which contains the robot's joint states.
4. Calculates the L2 norm of the difference between consecutive frames for the joint states. If the difference is greater than `0.015`, the robot is considered "moving" for that frame.
5. Calculates the stationary and moving durations for each episode and the overall dataset.
6. Checks for `sarm_progress.parquet` and summarizes the `progress_dense` scores (min, max, mean, and a sample from the first episode) if the file exists.
