# 夾爪抓取問題診斷與修復嘗試 - 工作日誌

**日期**: 2026-01-14 ~ 2026-01-15

## 問題描述

`pick_box` 和 `so101_pick_box` 環境中，夾爪無法正常抓取 Box，物體會彈開或穿透。
而 `pick_and_place` 環境可以正常夾取。

---

## 診斷過程

### 發現 1：環境設計目的不同

**關鍵發現：`pick_box_env` 是設計用來操作布料 (cloth) 的環境，不是抓取 Box！**

證據：
- 場景包含 `mujoco3_cloth.xml` 布料模擬
- `step()` 函數中有大量 cloth weld constraint 邏輯
- Debug 輸出顯示的 `cloth` 座標是布料節點（z 軸為負數且極大）
- `pick_and_place_env` 完全沒有布料相關代碼，純粹依賴物理摩擦

### 發現 2：物理參數差異

| 參數 | pick_and_place (正常) | pick_box (問題) |
|------|----------------------|-----------------|
| timestep | 0.002 | 0.02 (10倍大) |
| solver | PGS | CG |
| Box size | 0.025³ | 0.015³ |
| 夾爪摩擦力 | 7.0 | 無明確設定 |

### 發現 3：物理步進方式不同

```python
# pick_and_place (正確)
for _ in range(n_steps):
    robot.move_cartesian(Ti)
    mujoco.mj_step(model, data)  # 每步更新控制

# pick_box (問題)
mujoco.mj_step(model, data, n_steps)  # 批次執行，不更新控制
```

---

## 嘗試的修復方案

### 方案 A：修改 XML 物理參數

**修改內容：**
1. `timestep`: 0.02 → 0.002
2. `solver`: CG → PGS
3. Box 新增 `friction="7.0 1.0 0.5"`, `solimp`, `solref`
4. Box size: 0.015 → 0.025

**結果：❌ 仍然無法夾取**

### 方案 B：修改 step 函數為逐步執行

**修改內容：**
```python
for _ in range(n_steps):
    self._robot.move_cartesian(Ti)
    self._mj_data.ctrl[:6] = joint_position
    mujoco.mj_step(self._mj_model, self._mj_data)
```

**結果：❌ 仍然無法夾取，且有其他錯誤**

### 方案 C：創建純物理夾取環境 `pick_box_only_env.py`

**修改內容：**
- 基於 `pick_box_env.py` 創建簡化版本
- 移除所有布料相關邏輯
- Debug 輸出改為顯示與 Box 的距離

**結果：⚠️ 環境可啟動，但測試中仍有問題**

### 方案 D：回到舊版本 (90153e5) 檢查

**使用命令：**
```bash
git stash
git checkout 90153e535d9e0a90937916c83f5c7b648305ca01
# 測試後發現舊版本場景也有問題（路徑錯誤、缺少布料 XML）
git checkout -- .
git checkout -
git stash pop
```

**結果：❌ 舊版本無法直接運行，需要修改路徑**

---

## 創建的新檔案

1. `imitation_learning_lerobot/envs/pick_box_only_env.py` - 純物理夾取環境
2. `imitation_learning_lerobot/teleoperation/keyboard/pick_box_only_keyboard_handler.py` - 對應 handler

---

## 待解決問題

1. Box 仍會彈開，可能需要更深入調整接觸參數
2. 考慮使用 `pick_and_place` 環境作為 Box 夾取的基礎
3. SO101 夾爪的簡化碰撞幾何需要精確調整位置

---

## 關於 Git Submodule (lerobot) 的發現

在 `git checkout` 過程中發現 lerobot submodule 狀態不一致：
- 父專案記錄：`b464d9f8`（幾個月前）
- 實際版本：`fc296548`（最新 main）

**原因**：更新 lerobot 後沒有在父專案 commit submodule 版本變更

**解決方式**：
```bash
git add lerobot
git commit -m "Update lerobot submodule"
```
