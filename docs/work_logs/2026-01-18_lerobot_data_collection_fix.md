# [2026-01-18] LeRobot 資料蒐集腳本修復與分析

## 問題概述
修復 `collect_data.py` 腳本的 API 相容性問題，並分析資料集的動靜態比例。

---

## 1. LeRobot API 更新 (`add_frame`)

### 問題
```
TypeError: LeRobotDataset.add_frame() got an unexpected keyword argument 'task'
```

### 原因
新版 LeRobot API 不再接受 `task=` 關鍵字參數。

### 解決方案
將 `task` 放入 frame 字典中：
```python
# 舊版 API
dataset.add_frame(frame, task=task)

# 新版 API
frame["task"] = task
dataset.add_frame(frame)
```

**修改檔案**: `imitation_learning_lerobot/scripts/collect_data.py`

---

## 2. Renderer 被關閉錯誤

### 問題
```
RuntimeError: render cannot be called after close.
```

### 原因
`run()` 函數結尾呼叫 `self.close()`，但多 episode 時會重複使用同一個 env，導致 renderer 被關閉。

### 解決方案
移除 `run()` 中的 `self.close()`，改由調用方在結束時呼叫。

**修改檔案**: `imitation_learning_lerobot/envs/pick_and_place_env.py`

---

## 3. 手臂位置不復歸

### 問題
多 episode 時手臂不會回到初始位置。

### 原因
`_soft_reset_objects_and_time()` 缺少運動學設定。

### 解決方案
補齊以下設定（與 `reset()` 一致）：
- `self._robot.disable_base()`
- `self._robot.disable_tool()`
- `self._robot.set_base(...)`
- `self._robot.set_tool(sm.SE3.Trans(0.0, 0.0, 0.15))`

**修改檔案**: `imitation_learning_lerobot/envs/pick_and_place_env.py`

---

## 4. 新增 `--display` 參數

### 功能
讓使用者可選擇是否顯示 MuJoCo 視覺化介面。

### 使用方式
```bash
# 不顯示介面（預設）
python ./imitation_learning_lerobot/scripts/collect_data.py \
  --env.type=pick_and_place --episode=100

# 顯示介面
python ./imitation_learning_lerobot/scripts/collect_data.py \
  --env.type=pick_and_place --episode=100 --display
```

---

## 5. 資料集分析結果

**資料集路徑**: `outputs/datasets/pick_and_place`

| 統計項目 | 數值 |
|---------|------|
| 總幀數 | 681 |
| 移動中 | 573 (84.3%) |
| 靜止 | 107 (15.7%) |
| Episode 長度 | 227 幀 (固定) |

### 關於固定長度
程式使用**固定時間軌跡規劃**（總計 9.04 秒），不論方塊位置遠近都執行相同時間。

---

## 修改檔案清單

| 檔案 | 變更類型 |
|-----|---------|
| `imitation_learning_lerobot/scripts/collect_data.py` | 修改 |
| `imitation_learning_lerobot/envs/pick_and_place_env.py` | 修改 |


