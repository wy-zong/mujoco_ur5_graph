---
name: check-lerobot-version-diff
description: 比較本地 lerobot fork 與官方 huggingface/lerobot 的版本落差，列出目前版本號、落後幾個 commit、以及所有差異 commit（含 PR 號）。
---

# Check LeRobot Version Diff

此 skill 用來快速分析你本地 lerobot 與官方最新版本的差距，包含版本號、commit 數量、功能分類摘要。

## 前置條件

- 本地 `lerobot` 目錄已設定 `origin` remote 指向 `https://github.com/huggingface/lerobot.git`
- 有網路連線可以 `git fetch`

確認方式：
```powershell
cd c:\Users\ccu\mujoco_ur5_graph\lerobot
git remote -v
```
應看到 `origin  https://github.com/huggingface/lerobot.git`

## 步驟

### 1. 進入 lerobot 目錄

// turbo
```powershell
cd c:\Users\ccu\mujoco_ur5_graph\lerobot
```

### 2. 確認目前本地版本

// turbo
```powershell
git log --oneline -1
```

同時讀取 `pyproject.toml` 中的版本號：

// turbo
```powershell
Select-String -Path pyproject.toml -Pattern '^version'
```

### 3. 從官方抓取最新狀態

// turbo
```powershell
git fetch origin
```

### 4. 查看官方最新版本號與最新 tags

// turbo
```powershell
git tag --sort=-version:refname | Select-Object -First 5
```

// turbo
```powershell
git log --oneline origin/main -5
```

### 5. 計算落差：官方比你多的 commits

// turbo
```powershell
git log --oneline HEAD..origin/main
```

輸出格式為：
```
<commit_hash> <commit_message> (#PR號)
```

> commit message 末尾的 `(#XXXX)` 就是對應的 GitHub PR 號碼，可直接用以下 URL 查看詳情：
> `https://github.com/huggingface/lerobot/pull/XXXX`

### 6. 統計落後幾個 commit

// turbo
```powershell
git rev-list --count HEAD..origin/main
```

### 7. 分析差異重點（由 AI 整理）

拿到上述 commit 清單後，根據以下分類整理重點：

| 分類 | commit message 關鍵字 |
|------|----------------------|
| 🆕 新功能 | `feat` |
| 🐛 修復 | `fix` |
| ♻️ 重構 | `refactor` |
| 📦 依賴更新 | `chore(dependencies)`, `chore(deps)` |
| 📝 文件 | `chore(docs)`, `chore(readme)`, `chore(docstrings)` |
| ✅ 測試 | `test` |

## 輸出範例

```
目前版本：v0.4.5（commit 095856b0）
官方最新：v0.5.1（commit d9ec3a6f）
落後：38 個 commits

重要差異：
🆕 feat(robots): Unitree G1 WBC (#2876)
🆕 feat(train): cudnn_deterministic option (#3102)
🐛 fix(policies): crop losses based on action dof (#3133)
♻️ refactor(dataset): modular files (#3171)
📦 chore(dependencies): bump transformers v5 (#2964)
⚠️  feat(dependencies): require Python 3.12+ (#3023)  ← 破壞性變更
```

## 注意事項

- **破壞性變更**：特別留意 `feat(dependencies)` 類型的 commit，可能影響 Python 版本或套件相容性
- **transformers 版本**：升版前確認與現有 CLIP/VLM 模型的相容性
- **dataset 重構**：若 import 路徑改變，自訂腳本可能需要更新
