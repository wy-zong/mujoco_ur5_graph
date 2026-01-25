# MuJoCo UR5 Graph 專案深度解析

本文件詳細描述 `mujoco_ur5_graph` 專案的架構、功能實作細節以及與 `README.md` 的對照分析。

## 1. 專案概觀

本專案是一個結合 **MuJoCo 物理模擬** 與 **Hugging Face LeRobot** 框架的機器人模仿學習環境。核心目標是提供一個穩定的雙手臂（UR5e + SO101）協作模擬環境，用於收集訓練數據以訓練 ACT、SmolVLA 等機器人控制模型。

### 核心特性
- **雙手臂協作**：同時模擬工業級 UR5e 機械手臂與低成本開源 SO101 機械手臂。
- **混合控制模式**：UR5e 使用傳統直角座標控制 (Cartesian)，SO101 使用圓柱座標系控制 (Cylindrical) 以適應其運動學特性。
- **高效渲染**：支援 Windows 下的 GLFW 後端 GPU 加速渲染。
- **模仿學習整合**：直接生成符合 LeRobot 格式的數據集（Images + States + Actions）。

---

## 2. 專案結構深度解析

```
mujoco_ur5_graph/
├── imitation_learning_lerobot/   # 核心應用邏輯包
│   ├── envs/
│   │   ├── pick_box_env.py       # [核心] 雙手臂抓取環境 (UR5e + SO101)
│   │   └── pick_and_place_env.py # [測試] 單手臂 UR5e 測試環境
│   ├── teleoperation/
│   │   └── keyboard/             # 鍵盤控制實作
│   │       ├── pick_box_keyboard_handler.py # [核心] 雙手臂鍵盤控制器
│   │       └── ...
│   └── arm/                      # 機器人運動學定義 (Robot, UR5e, SO101)
├── docs/                         # 文檔存放區
├── scripts/                      # 工具腳本
├── tests/                        # 測試與驗證腳本
│   ├── run_pick_and_place.py     # 單臂環境啟動腳本
│   └── test_so101_env.py         # SO101 測試腳本
└── outputs/                      # 輸出資料（忽略於 git）
    └── motor_dynamics/           # 馬達動態數據
```

---

## 3. 環境與動作空間詳解

### 核心環境：`PickBoxEnv` (`pick_box_env.py`)
這是本專案的主要工作環境，支援雙手臂協作。

#### 狀態空間 (Observation Space)
- **Pixels (Images)**:
  - `top`: 俯視攝影機 (Global view)
  - `hand`: 手部攝影機 (Eye-in-hand)
- **Agent Pos (State)**: 包含機械手臂末端位置與夾爪狀態。

#### 動作空間 (Action Space) - 14維
這一部分在代碼實作上與 `README.md` 略有差異，以下為**實際代碼 (`pick_box_env.py`) 的定義**：

| Index | 機器人 | 控制量 | 實作邏輯 (Code) | 備註 |
|-------|--------|--------|----------------|------|
| 0-2   | UR5e   | Pos    | `dx`, `dy`, `dz` | 直角座標增量 |
| 3     | UR5e   | Gripper| `0.0` (開) - `1.0` (關) | 轉為 0-255 控制訊號 |
| 4-6   | UR5e   | Rot    | `roll`, `pitch`, `yaw` | 尤拉角增量 |
| **7**     | **SO101** | **Pos** | **`dr` (半徑增量)** | **圓柱座標系** (非直角座標 x) |
| **8**     | **SO101** | **Pos** | **`dtheta` (角度增量)** | **圓柱座標系** (非直角座標 y) |
| 9     | SO101  | Pos    | `dz` (高度增量) | 直角座標 z |
| 10    | SO101  | Rot    | `droll` | 手腕旋轉 (Roll) |
| 11    | SO101  | Rot    | `dpitch` | 手腕俯仰 (Pitch) |
| 12    | SO101  | Gripper| `0.0` (開) - `1.0` (關) | 正規化夾爪控制 |
| 13    | SO101  | (Res)  | 保留/未使用 | 程式碼中存在但未被用於邏輯 |

> **⚠️ 重要發現**：`README.md` 中描述 SO101 的動作為 `dx, dy, dz`，但原始碼 (`pick_box_env.py` line 342-370) 實際上實作了 **圓柱座標控制 (Radius, Theta, Z)**。這是為了讓 SO101 這類手臂更容易操作（前後伸縮 + 左右旋轉），而非傳統的 XY 平移。

### 測試環境：`PickAndPlaceEnv` (`pick_and_place_env.py`)
這是一個簡化的單手臂環境，僅包含 UR5e。
- **Action Dim**: 4 (dx, dy, dz, gripper) - 僅支援位置控制，無旋轉。
- **用途**: 用於測試基礎物理模擬與渲染功能 (`tests/run_pick_and_place.py`)。

---

## 4. 控制系統 (`teleoperation`)

鍵盤控制邏輯位於 `imitation_learning_lerobot/teleoperation/keyboard/pick_box_keyboard_handler.py`。

### 雙手臂切換機制
- 使用 `0`, `.`, `Tab`, 或 `Delete` 鍵在 **UR5e (Active=0)** 與 **SO101 (Active=1)** 之間切換。
- 顯示訊息：`[INFO] Active arm: UR5` 或 `[INFO] Active arm: SO101`。

### 按鍵映射 (實際代碼行為)

#### UR5e 模式
- **移動 (XYZ)**: 數字鍵 `8/2` (+Z/-Z), `4/6` (+X/-X), `7/1` (+Y/-Y)
- **旋轉 (RPY)**:
    - Roll: `/` / `*`
    - Pitch: `-` / `+`
    - Yaw: `5` / `0` (或 Insert)
- **夾爪**: `9` (關), `3` (開)

#### SO101 模式 (對應上述圓柱座標)
- **前後伸縮 (Radius)**: 對應按鍵的 `X` 軸輸入 (`1`/`7` 或 `End`/`Home`) → 對應 `action[7]` (`dr`)
- **左右旋轉 (Theta)**: 對應按鍵的 `Y` 軸輸入 (`4`/`6` 或 `Left`/`Right`) → 對應 `action[8]` (`dtheta`)
- **上下移動 (Z)**: 對應按鍵的 `Z` 軸輸入 (`8`/`2`) → 對應 `action[9]` (`dz`)
- **手腕姿態**:
    - Roll: `/` / `*`
    - Pitch: `-` / `+`
- **夾爪**: `9` / `3`

---

## 5. 技術細節與優化

1.  **MuJoCo GPU 渲染**:
    - 腳本中透過 `os.environ["MUJOCO_GL"] = "glfw"` 強制啟用 GLFW 後端，這對 Windows 上的高效能渲染至關重要。

2.  **IK 逆運動學**:
    - UR5e 使用 `roboticstoolbox` 或自定義的解析解/數值解 (基於 `spatialmath`)。
    - SO101 使用 `lerobot.model.kinematics.RobotKinematics` (基於 K-L 散度或 DLS 方法) 進行逆運動學計算，並在 `pick_box_env.py` 中加入了針對圓柱座標的目標位置 latching 機制，以穩定控制。

3.  **物體吸附 (Grasping)**:
    - 實作了 `mujoco.mj_attach` 機制，透過 Equality Constraint (Weld) 來模擬穩定的抓取，避免物理引擎中單純靠摩擦力抓取不穩定的問題。
    - `PickBoxEnv` 中包含了 `detach` 邏輯，允許夾爪張開時釋放物體。

## 6. 總結

本專案是一個功能完整的雙臂模擬環境。使用者應特別注意 **SO101 的控制策略是基於圓柱座標系**，這與傳統直角座標系不同，操作直覺上是「手臂伸縮」與「底座旋轉」，而非前後左右平移。這一設計更符合 SO101 這類 Scara-like 或自定義構型手臂的操作邏輯。
