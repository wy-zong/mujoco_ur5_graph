# SO101 夾爪控制與夾取功能除錯紀錄

## 1. 夾爪控制方向修正
- **問題**: 夾爪控制方向相反（9/PageUp 變打開，3/PageDown 變關閉）。
- **修正**: 修改 `so101_pick_box_env.py` 中的控制邏輯。
    - 定義 `action[5]`: -1 為關閉，+1 為打開。
    - 修正 `np.clip` 範圍為 `[-1, 1]` (原為 `[0, 1]` 導致無法關閉)。
    - 對應 MuJoCo ctrl 值：0.0 (閉) -> -0.5, 1.0 (開) -> 1.5。

## 2. 方塊位置調整
- **問題**: 方塊初始位置超出 SO101 工作範圍。
- **修正**:
    - 修改 `so101_pick_box_scene.xml`: 設定 `pos="1.70 0.6 0.78"`。
    - 修正 `so101_pick_box_env.py`: 註解掉 `reset()` 中的隨機化位置代碼，確保 XML 設定生效。

## 3. 夾爪初始狀態
- **調整**: 修改 `so101_pick_box_env.py`，將 `self._gripper_state` 初始化為 `0.0` (閉合狀態)。

## 4. 夾取功能除錯 (摩擦力 vs Weld)
- **症狀**: 夾爪能閉合接觸方塊，但無法利用摩擦力夾起方塊 (方塊滑落)。
- **分析與嘗試**:
    1. **碰撞參數檢查**: 發現 `so101_collision` class 缺少 `contype/conaffinity`，導致可能無碰撞。已補上 `contype="3" conaffinity="3"` 以匹配 Box。
    2. **增加摩擦力**:
        - 修改 `so101_new_calib.xml` (夾爪) 與 `so101_pick_box_scene.xml` (Box)。
        - 摩擦力參數調升至 `friction="7.0 1.0 0.5"`。
        - 增加 `impratio="10"` (參考 2F85 設定) 以提升接觸穩定性。
        - 設定 `priority="1"` 與更硬的 `solref/solimp`。
    3. **對照組分析 (UR5e + 2F85)**:
        - 確認 `pick_and_place` 環境 (UR5e) 是靠純摩擦力成功夾取。
        - 關鍵差異: 2F85 為雙側平行夾爪 (Double Jaw)，SO101 為單側活動夾爪 (Single Jaw)。
        - SO101 的非對稱結構在 MuJoCo 中較難產生穩定的夾持力。

## 5. 目前狀態
- 夾爪控制邏輯正常。
- 碰撞與摩擦力參數已最佳化 (參考工業級 2F85 設定)。
- 仍遺留物理夾取不穩定的問題，可能需考慮結構限制或改用 Weld Constraint 模擬。
