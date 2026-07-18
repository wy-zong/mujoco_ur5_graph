# 在「接手臂 USB 的 Windows 執行主機」上執行。
# 建議用「系統管理員 PowerShell」。

# 1. 安裝 usbipd-win。
# 如果已經安裝可略過。
winget install --interactive --exact dorssel.usbipd-win

# 2. 確認版本與服務。
usbipd --version
Get-Service usbipd

# 3. 列出 USB 裝置。
# 先看目前列表，再用拔插 COM6/COM8/COM4/COM9 的方式找出每個 USB adapter 的 BUSID。
usbipd list

# 4. 依實際 BUSID 修改後執行。
# 範例：
# usbipd bind --busid 4-1
# usbipd bind --busid 4-2
# usbipd bind --busid 4-3
# usbipd bind --busid 4-4

# COM6 / left follower 的 BUSID：
usbipd bind --busid TODO_COM6_BUSID

# COM8 / right follower 的 BUSID：
usbipd bind --busid TODO_COM8_BUSID

# COM4 / left leader 的 BUSID：
usbipd bind --busid TODO_COM4_BUSID

# COM9 / right leader 的 BUSID：
usbipd bind --busid TODO_COM9_BUSID

# 5. 再次確認 STATE 是 Shared。
usbipd list

# 6. 確認執行主機 IP，推理主機 WSL 會用這個 IP 連線。
ipconfig

# 注意：
# usbipd-win 使用 TCP 3240。安裝時通常會建立 usbipd 防火牆規則。
# 如果推理主機 WSL 無法連線，先檢查 Windows 防火牆是否允許 TCP 3240。
