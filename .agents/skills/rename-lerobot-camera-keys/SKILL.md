---
name: rename-lerobot-camera-keys
description: Rename LeRobot camera/video feature keys across info.json, stats.json, data parquet, meta/episodes parquet, and videos folder paths. Use when users need camera key renaming such as left_camera1/3 to camera1/3.
---

# Rename LeRobot Camera Keys

Use this skill when a LeRobot dataset needs camera key renaming while keeping dataset consistency.

## What this updates

- `meta/info.json` feature keys
- `meta/stats.json` keys (if present)
- `data/chunk-*/file-*.parquet` column names
- `meta/episodes/chunk-*/file-*.parquet` columns under `videos/<key>/*`
- `videos/<key>/...` directory names

## Preferred workflow

1. Always write to a new dataset folder first.
2. Run the bundled script in `new_mj` environment.
3. Validate with `--verify-only`.

## Command

```powershell
conda run -n new_mj python .agents/skills/rename-lerobot-camera-keys/scripts/rename_lerobot_camera_keys.py \
  --src "<source_dataset_dir>" \
  --dst "<new_dataset_dir>"
```

## Optional mapping override

Default mapping:

- `observation.images.left_camera1` -> `observation.images.camera1`
- `observation.images.left_camera3` -> `observation.images.camera3`

Override with JSON:

```powershell
conda run -n new_mj python .agents/skills/rename-lerobot-camera-keys/scripts/rename_lerobot_camera_keys.py \
  --src "<source_dataset_dir>" \
  --dst "<new_dataset_dir>" \
  --mapping-json '{"observation.images.left_camera1":"observation.images.camera1","observation.images.left_camera3":"observation.images.camera3"}'
```

## Verification

```powershell
conda run -n new_mj python .agents/skills/rename-lerobot-camera-keys/scripts/rename_lerobot_camera_keys.py \
  --src "<source_dataset_dir>" \
  --dst "<new_dataset_dir>" \
  --verify-only
```