# PickBoxEnv 與 LeRobot EnvHub 整合報告

## 📋 專案概述

本文件記錄 PickBoxEnv MuJoCo 模擬環境與 LeRobot EnvHub 的整合工作進度。

---

## ✅ 已完成工作

### 1. 跨平台相容性修正

| 檔案 | 修改內容 |
|------|----------|
| `pick_box_env.py` | URDF 路徑從硬編碼 Linux 路徑改為相對路徑 |
| `collect_data_teleoperation.py` | 添加 `sys.path` 設定以支援跨平台 import |
| `pick_box_keyboard_handler.py` | 添加 Windows 替代按鍵支援 |

### 2. Windows 鍵盤控制支援

新增替代按鍵對應表：

| 功能 | 數字鍵盤 | 替代按鍵 |
|------|---------|---------|
| +X | `4` | `←` Left |
| -X | `6` | `→` Right |
| +Y | `7` | Home |
| -Y | `1` | End |
| +Z | `8` | `↑` Up |
| -Z | `2` | `↓` Down |
| Gripper 開 | `3` | Page Down |
| Gripper 關 | `9` | Page Up |

### 3. EnvHub 套件結構（本地測試完成）

```
envhub/pickbox-env/
├── env.py                 # LeRobot make_env 入口點
├── pickbox_gym_env.py     # Gymnasium 封裝類別
├── requirements.txt       # 依賴清單
├── README.md              # 使用說明
├── test_interactive.py    # 互動式測試腳本
├── test_keyboard.py       # 鍵盤測試腳本
└── assets/
    ├── scenes/
    │   └── pick_box_scene_copy.xml
    └── SO-ARM100/
        └── Simulation/SO101/so101_new_calib.urdf
```

### 4. Git 提交

- **Commit:** `87d9708`
- **遠端:** https://github.com/wy-zong/mujoco_ur5_graph

---

## 🔄 後續 EnvHub 規劃

### 階段一：完善本地套件（待執行）

**目標：** 讓 EnvHub 套件可以完全獨立運行。

**待辦事項：**
1. 將 `PickBoxEnv` 完整程式碼複製到 `envhub/pickbox-env/` 中
2. 修正所有內部依賴路徑
3. 確保資源檔案完整（XML, URDF, mesh 等）
4. 測試獨立環境中是否能正常運行

### 階段二：上傳 HuggingFace Hub

**目標：** 讓其他使用者可以透過 LeRobot 直接載入環境。

**上傳步驟：**
```bash
# 1. 登入 HuggingFace
huggingface-cli login

# 2. 創建 repo 並上傳
huggingface-cli upload <username>/pickbox-env ./envhub/pickbox-env --repo-type space
```

**使用方式：**
```python
from lerobot.envs.factory import make_env

envs = make_env("username/pickbox-env", n_envs=1, trust_remote_code=True)
suite = next(iter(envs))
env = envs[suite][0]
obs, info = env.reset()
```

### 階段三：與 LeRobot 訓練流程整合

**目標：** 使用 LeRobot CLI 進行資料收集和模型訓練。

**可能的整合點：**
1. `lerobot-record` - 資料收集
2. `lerobot-train` - 模型訓練
3. `lerobot-eval` - 模型評估

---

## ⚠️ 已知問題

### 1. NumPy 版本衝突

`roboticstoolbox` 需要 NumPy < 2.0，但部分套件（如 `rerun-sdk`）需要 NumPy >= 2.0。

**暫時解決方案：**
```bash
pip install numpy==1.26.4
```

### 2. OpenCV GUI 支援

部分環境安裝的是 `opencv-python-headless`，不支援 GUI 視窗。

**解決方案：**
```bash
pip uninstall opencv-python-headless -y
pip install opencv-python
```

---

## 📁 相關檔案路徑

| 項目 | 路徑 |
|------|------|
| 專案根目錄 | `D:/lerobot_arm/mujoco_ur5_graph` |
| PickBoxEnv | `imitation_learning_lerobot/envs/pick_box_env.py` |
| 鍵盤 Handler | `imitation_learning_lerobot/teleoperation/keyboard/pick_box_keyboard_handler.py` |
| 資料收集腳本 | `imitation_learning_lerobot/scripts/collect_data_teleoperation.py` |
| EnvHub 套件 | `envhub/pickbox-env/` |

---

## 📅 更新記錄

| 日期 | 內容 |
|------|------|
| 2026-01-13 | 完成跨平台相容性修正，新增 Windows 鍵盤支援，創建 EnvHub 套件結構 |
| 2026-01-14 | GPU 渲染優化，控制頻率調整為 30Hz 匹配真機 |

---

## 🚀 性能優化（2026-01-14）

### 1. GPU 渲染啟用

**問題：** MuJoCo 預設使用 CPU 軟體渲染，導致每步延遲 ~500ms。

**解決方案：**
1. 在 `pick_box_env.py` 開頭設定環境變數：
   ```python
   os.environ["MUJOCO_GL"] = "glfw"
   ```
2. 在 Windows 設定中強制 Python 使用獨立顯卡（NVIDIA 控制面板）

**效果：** 渲染時間從 ~474ms 降到 ~25-35ms（提升 15-20 倍）

### 2. 控制頻率調整

| 參數 | 舊值 | 新值 | 說明 |
|------|------|------|------|
| `sim_hz` | 500 | 600 | 物理模擬頻率 |
| `control_hz` | 25 | **30** | 匹配真機 30fps 相機 |
| `n_steps` | 20 | 20 | 每控制步物理模擬次數 |

### 3. 性能分析工具

在 `step()` 中添加計時器：
```
[PERF] physics: 46ms, logic: 1ms, obs: 1ms, total: 48ms
```

---

## ⚙️ 環境設定要求

### Windows GPU 渲染設定

1. **系統設定 → 顯示 → 圖形**
2. 新增 `python.exe`，選擇「高效能」
3. 或透過 NVIDIA 控制面板設定

### 依賴版本

```
numpy==1.26.4
mujoco>=3.0
opencv-python (非 headless 版本)
```
