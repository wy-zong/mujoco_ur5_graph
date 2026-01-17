# Git 常用指令學習筆記

## 版本切換與檢查

### 暫存當前修改 (Stash)
```bash
# 暫存所有未提交的修改
git stash

# 暫存並添加說明
git stash push -m "工作說明"

# 恢復暫存的修改
git stash pop

# 查看暫存列表
git stash list
```

### 切換到特定 Commit
```bash
# 切換到特定 commit (進入 detached HEAD 狀態)
git checkout <commit-hash>

# 例如
git checkout 90153e535d9e0a90937916c83f5c7b648305ca01

# 回到之前的分支
git checkout -

# 或指定分支名
git checkout main
```

### 放棄本地修改
```bash
# 放棄單一檔案的修改
git checkout -- <file>

# 放棄所有修改
git checkout -- .
```

---

## Submodule 操作

### 什麼是 Submodule？
- Git 內嵌另一個 Git 儲存庫的方式
- 父專案只記錄 submodule 的 commit hash（像書籤）
- 常用於引用外部函式庫

### 常用指令
```bash
# 查看 submodule 狀態
git submodule status

# 初始化並更新 submodule
git submodule update --init

# 查看父專案記錄的 submodule commit
git ls-tree HEAD <submodule-name>

# 查看 submodule 實際的 commit
git -C <submodule-dir> rev-parse HEAD

# 查看 submodule 內部狀態
git -C <submodule-dir> status
```

### 更新 Submodule 版本記錄
```bash
# 在 submodule 內更新後，需要在父專案 commit
git add <submodule-name>
git commit -m "Update submodule"
```

---

## 查看歷史記錄

```bash
# 查看最近 N 個 commit
git log --oneline -n 5

# 查看特定檔案的修改歷史
git log --oneline -5 -- <file>

# 查看特定 commit 中某檔案的內容
git show <commit>:<file>
```

---

## 分支操作

```bash
# 創建新分支
git checkout -b <branch-name>

# 或使用 switch (較新的語法)
git switch -c <branch-name>

# 切換分支
git checkout <branch-name>
git switch <branch-name>
```

---

## 常見狀態說明

### Detached HEAD
- 當 checkout 到特定 commit 時出現
- 可以查看和測試，但不建議直接 commit
- 離開時用 `git checkout <branch>` 回到正常狀態

### Submodule 狀態符號
- `-` : Submodule 未初始化
- `+` : Submodule 指向不同的 commit
- 無符號 : 正常

### 常見錯誤訊息
```bash
# "modified content" in submodule
# 表示 submodule 內有未提交的修改

# "new commits" in submodule  
# 表示父專案記錄的版本與 submodule 實際版本不同
```

---

## 實用組合技

### 臨時測試舊版本後恢復
```bash
git stash                    # 1. 暫存當前修改
git checkout <old-commit>    # 2. 切換到舊版本測試
# ... 測試 ...
git checkout -- .            # 3. 放棄測試中的修改
git checkout -               # 4. 回到原分支
git stash pop                # 5. 恢復暫存的修改
```

### 只恢復特定檔案到舊版本
```bash
git checkout <commit> -- <file>
```
