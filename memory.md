# 分析與改動紀錄：解決 async_inference 的 Dynamixel 封包遺失問題

## 問題描述
在執行 `python -m lerobot.async_inference.robot_client` 進行遠端推論（Remote Inference）時，發現執行一段時間後會出現以下錯誤：

```
ERROR 2026-04-28 13:57:36 t_client.py:456 Error in observation sender: Failed to sync read 'Present_Position' on ids=[1, 2, 3, 4, 5, 6] after 1 tries. [TxRxResult] There is no status packet!
```

## 根本原因分析 (Root Cause Analysis)

### 第一階段分析（gRPC 阻塞）
一開始發現 gRPC 發送設定後，等待伺服器準備模型的時間長達 1 分鐘以上，在這期間 Dynamixel 已經連線並被閒置，進而導致 USB 進入超時斷線的狀態。

### 第二階段分析（相機暖機阻塞）
在解決 gRPC 阻塞後，發現問題依然存在。仔細觀察 log 發現，當客戶端嘗試連線硬體時：
- 每個 OpenCVCamera 的連線與暖機需要約 4 秒。
- 有 3 個相機，總共耗時約 12 秒。
原本的程式碼順序會先連線 Dynamixel bus (馬達)，接著才循序連線各個相機。這意味著即使延後了 `robot.connect()`，馬達連線後仍會被相機的暖機過程卡住長達 12 秒。在 Windows 的機制下（如 USB Selective Suspend 或 Serial Port timeout），這 12 秒的無封包傳輸時間已經足以導致通訊埠掉線。

## 實際改動 (Changes Made)

### 1. 將 `robot.connect()` 移至 gRPC 準備完成後
避免伺服器載入模型的長達一分鐘閒置期導致斷線。
**檔案：** `lerobot/src/lerobot/async_inference/robot_client.py`
移除 `RobotClient.__init__` 中的 `self.robot.connect()`，並將它移動到 `RobotClient.start` 之中：
```python
            self.stub.SendPolicyInstructions(policy_setup)

            self.logger.info("Connecting to robot hardware...")
            self.robot.connect() # <-- 新增到這裡
```

### 2. 調整硬體連線順序（先相機，後馬達）
避免相機暖機的 12 秒等待期間，讓提早連線的 Dynamixel bus 處於閒置斷線狀態。
**檔案一：** `lerobot/src/lerobot/robots/so_follower/so_follower.py`
將相機的連線移動到 `self.bus.connect()` 前方：
```python
    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        for cam in self.cameras.values():
            cam.connect()
        self.bus.connect()
        # ...
```

**檔案二：** `lerobot/src/lerobot/robots/bi_so_follower/bi_so_follower.py`
將頂層相機的連線移動到左右手臂的前方：
```python
    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        for cam in self.top_cameras.values():
            cam.connect()
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)
```

這些改動確保了最容易發生 Timeout 的馬達通訊埠，總是在「所有耗時阻塞（下載模型、相機暖機）」都結束，且即將開始高頻率控制迴圈的前一刻，才真正被打開連線，完全杜絕任何閒置導致的斷線可能。