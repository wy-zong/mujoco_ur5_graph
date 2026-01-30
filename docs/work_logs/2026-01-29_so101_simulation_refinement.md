# [2026-01-29] SO101 模擬環境重置與路徑優化紀錄

## 摘要
本日針對 SO101 手臂在 MuJoCo 模擬環境中的初始狀態定義、重置行為一致性以及自動夾取腳本 (`run`) 的軌跡平滑度進行了修正與優化。重點解決了重置時的視覺跳動問題，並確認了夾爪插值行為的物理特性。

## 1. 初始姿勢與重置一致性修正 (Initial Pose & Reset Consistency)

### 問題 (Problem)
- 手臂在 `reset()` 後與 `run()` 結束後的姿勢不一致，導致下一輪開始時有明顯的視覺跳動。
- `HOME_Q` 設定值中，Shoulder Lift 關節角度 (`-1.74664`) 超出物理極限 (`-1.74533`)，導致物理引擎介入修正時產生瞬間彈跳。

### 分析與嘗試 (Analysis & Attempts)
- **嘗試 1 (XML Keyframe)**: 在 XML 中定義 `<keyframe>`。
    - *結果*: 僅對 Viewer 啟動有效，無法解決 Python 腳本中的控制指令同步問題。
- **嘗試 2 (Physics Settling)**: 在 `reset()` 與 `_soft_reset` 中同步 `ctrl` 指令並加入 100 步的物理沉降 (`mj_step`)。
    - *結果*: 有效解決初始不穩，但與 `run()` 結束時的重力下垂狀態仍有微小差異。
- **嘗試 3 (Force Synchronization)**: 試圖在 `run()` 結束時強制將手臂「瞬間移動」回完美的 `HOME_Q` 狀態。
    - *結果*: 用戶撤回此改動，認為這是在逃避問題，應解決回位過程本身。

### 解決方案 (Solution)
- **修正 Joint Limits**: 微調 `HOME_Q` 的 Shoulder Lift 至 `-1.73664`，確保在物理極限內。
- **保留物理沉降**: 維持 `reset` 函數中的物理沉降步驟，確保每次重置都是物理穩定的。

## 2. 夾爪回位視覺卡頓問題 (Gripper "Stuck" Issue)

### 問題 (Problem)
- 用戶觀察到在「Return Home」階段，夾爪似乎比手臂先到位並卡住，看起來不自然。

### 分析 (Verification)
- 檢查 `move_to_joint_target` 函數，確認使用線性插值 (`Linear Interpolation`)。
- **數據分析**:
    - 手臂關節移動幅度：約 90 度。
    - 夾爪移動幅度：僅約 5 度 (`-5.0` -> `-9.9` (閉合))。
- **結論**: 數學上所有關節同時到達。但因為夾爪行程極短，每一步的變化量微乎其微，視覺上看起來就像是「已經到位靜止」，而手臂仍在大幅移動。
- *狀態*: 用戶確認此物理現象解釋合理，視為功能正常。

## 3. 軌跡平滑化 (Trajectory Smoothing)

### 問題 (Problem)
- 初始抬升 (`Raise`) 與回位 (`Return Home`) 動作生硬，且 IK 解算有時會產生不必要的翻轉。

### 解決方案 (Solution)
- **Raise Phase**: 加入 `target_orientation_weight=0.1` 進行弱約束，引導 IK 保持姿態平穩。
- **Return Phase**: 棄用 IK，改寫 `move_to_joint_target` 函數。
    - 直接在關節空間 (Joint Space) 進行線性插值。
    - 優點：絕對保證回到 `HOME_Q`，不會有 IK 多解或奇異點問題。

## 4. 腳本錯誤修正 (Script Fixes)

### 問題 (Problem)
- `collect_data.py` 執行報錯 `KeyError: 'handside'` 與 `NameError: name 'home_q' is not defined`。

### 解決方案 (Solution)
- **Sytnax Fix**: 在 `_cameras` 列表中補上遺漏的逗號 (`"hand", "side"`）。
- **Variable Fix**: 在 `run` 函數中正確定義並轉換 `home_q_rad` 至 `home_q_deg` 供 FK 計算使用。

## 當前狀態 (Current Status)
- 環境代碼 `so101_joint_control_env.py` 已穩定。
- 資料蒐集腳本 `collect_data.py` 可正常執行。
- 視覺跳動與回位行為已獲得解釋與驗證。
