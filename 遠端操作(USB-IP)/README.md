# 遠端操作 USB/IP 流程

目標：

- 執行主機：Windows，手臂 USB 實體接在這台。
- 推理主機：另一台 Windows 裡的 WSL，透過 SSH 進去執行 `lerobot-rollout`。
- USB/IP 方向：執行主機 Windows 分享 USB 裝置，推理主機 WSL attach 遠端 USB。

這條路不使用 `ssh -R 17000`、`socat`、`/tmp/lerobot_*`。成功後，推理主機 WSL 會直接看到 Linux 裝置，例如：

```bash
/dev/ttyACM0
/dev/ttyACM1
/dev/ttyUSB0
/dev/ttyUSB1
```

## 重要限制

- USB 裝置 attach 到 WSL 後，Windows 執行主機不能同時使用該 USB 裝置。
- 每一個 USB adapter 都要各自 `bind` 和 `attach`。
- attach 後的 `/dev/tty*` 編號可能每次不同，要用 `dmesg` 或 `ls -l /dev/serial/by-id/` 確認。
- 如果執行主機和推理主機不在同一個 LAN，Windows 防火牆或路由可能會擋 TCP 3240。

## 檔案

- `01_執行主機_windows_admin.ps1`：在接手臂的 Windows 上，以系統管理員 PowerShell 執行。
- `02_推理主機_wsl.sh`：SSH 到推理主機 WSL 後執行，用來安裝工具、list、attach、驗證。
- `03_rollout_usbip.sh`：attach 成功後，把 `/dev/tty*` 填進 rollout。

## 建議測試順序

1. 在執行主機 Windows 跑 `usbipd list`，找出 COM6/COM8/COM4/COM9 對應的 `BUSID`。
2. 在執行主機 Windows 對四個 `BUSID` 執行 `usbipd bind --busid <BUSID>`。
3. 在推理主機 WSL 跑 `usbip list --remote=<執行主機IP>`，確認能看到分享出來的裝置。
4. 在推理主機 WSL 跑 `sudo usbip attach --remote=<執行主機IP> --busid=<BUSID>`。
5. 在推理主機 WSL 用 `lsusb`、`dmesg | tail -80`、`ls -l /dev/ttyACM* /dev/ttyUSB*` 確認裝置節點。
6. 先只測 left follower 的單臂 LeRobot 連線，再跑完整 bimanual rollout。

## 參考

- Microsoft WSL USB/IP 文件：https://learn.microsoft.com/windows/wsl/connect-usb
- usbipd-win README：https://github.com/dorssel/usbipd-win
