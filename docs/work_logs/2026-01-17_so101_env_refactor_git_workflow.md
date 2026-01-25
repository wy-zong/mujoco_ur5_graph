# [2026-01-17] SO101 環境重構與 Git 上傳流程整理

## 概述

本次工作主要完成了 SO101 環境的重命名與重構，以及 Git 上傳流程的整理與修正。

---

## Git 改動報告生成

### 問題
- 用戶想查看自上次 git push 後的所有改動

### 解決方案
- 使用 `git diff --stat HEAD` 和 `git status` 查看改動
- 生成了詳細的改動報告，包含 13 個已修改檔案

---

## SO101 環境檔案分析與合併建議

### 發現
識別出兩個相似的 SO101 環境檔案：
- `so101_pick_box_env.py` - 正式版，末端點控制
- `so101_pick_box_env_test.py` - 測試版，新增直接關節控制

### 差異分析
Test 版本新增的功能：
- `_boost_controller_gains()` - 提高控制器增益
- `set_joint_positions()` - 直接關節角度控制（用於實機同步）

### 決策
用戶決定**不合併**，而是**重命名**測試版本以更清楚表達功能：
- `so101_pick_box_env_test.py` → `so101_pick_box_env_hybrid.py`
- `SO101PickBoxEnvTest` → `SO101PickBoxEnvHybrid`

---

## 環境重命名執行

### 修改的檔案
| 檔案 | 變更 |
|------|------|
| `so101_pick_box_env_hybrid.py` | 類別名稱和註釋更新 |
| `envs/__init__.py` | import 語句更新 |
| `hybrid_teleoperate_test.py` | import 和類型註解更新 |
| `test_sim_direct_control.py` | import 和所有引用更新 |

---

## Git 上傳問題與修正

### 問題
首次 `git add -A` 導致不想上傳的檔案（如 `envhub/`、`.agent/`、`MUJOCO_LOG.TXT`）被加入

### 原因
`.gitignore` 未包含這些目錄/檔案

### 解決方案
1. 使用 `git reset --soft` 撤回 commit（保留本地檔案）
2. 使用 `git push --force` 撤回遠端上傳
3. 清空暫存區 `git reset HEAD`
4. 手動 `git add` 指定檔案
5. 更新 `.gitignore` 添加忽略規則

### 更新的 `.gitignore`
```gitignore
# Project specific ignores
envhub/
.agent/
MUJOCO_LOG.TXT
```

---

## 最終上傳的檔案

### Commit 1: `c4a91da` - 核心環境 (17 個檔案)
- `so101_pick_box_env_hybrid.py` (新)
- `so101_pick_box_env.py` (新)
- `so101_pick_box_scene.xml` (新)
- `hybrid_teleoperate_test.py` (新)
- `so101_new_calib.xml` (改)
- `pick_box_scene_copy.xml` (改)
- 其他環境和配置檔案

### Commit 2: `d40b565` - 文檔和額外環境 (11 個檔案)
- `docs/` 工作日誌和專案說明
- `pick_box_only_env.py` (新)
- `pick_box_only_scene.xml` (新)
- 鍵盤處理器
- `.gitignore` (改)

---

## 學到的 Git 知識

| 指令 | 用途 |
|------|------|
| `git add -A -n` | 預覽將要添加的檔案（dry run） |
| `git add -p` | 互動式 patch 模式 |
| `git reset --soft <commit>` | 撤回 commit，保留暫存區和本地檔案 |
| `git reset HEAD` | 清空暫存區，不動本地檔案 |
| `git push --force` | 強制推送覆蓋遠端 |

---

## 當前狀態

✅ 已完成所有上傳  
✅ `.gitignore` 已更新，防止意外上傳  
✅ 環境命名已重構為 `Hybrid` 版本

## 下一步

- 測試 `SO101PickBoxEnvHybrid` 的功能
- 驗證混合遙操作腳本 `hybrid_teleoperate_test.py`
