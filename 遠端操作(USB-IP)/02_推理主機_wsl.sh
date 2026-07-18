# 在「推理主機 Windows 的 WSL」中執行。
# 這裡的 EXECUTION_HOST_IP 是接手臂 USB 的 Windows 執行主機 IP。

EXECUTION_HOST_IP=192.168.50.161

# 1. 確認 WSL kernel。
uname -a

# 2. 安裝 USB/IP client 工具與檢查工具。
sudo apt update
sudo apt install -y linux-tools-generic hwdata usbutils

# 3. 如果 usbip 指令不存在，用 linux-tools 裡的 usbip 建立替代連結。
if ! command -v usbip >/dev/null 2>&1; then
  USBIP_BIN="$(find /usr/lib/linux-tools -name usbip -type f | sort | tail -n 1)"
  if [ -n "$USBIP_BIN" ]; then
    sudo update-alternatives --install /usr/local/bin/usbip usbip "$USBIP_BIN" 20
  fi
fi

usbip version

# 4. 從推理主機 WSL 查詢執行主機 Windows 分享出來的 USB 裝置。
usbip list --remote="${EXECUTION_HOST_IP}"

# 5. 依實際 BUSID 修改後 attach。
# 範例：
# sudo usbip attach --remote="${EXECUTION_HOST_IP}" --busid=4-1
# sudo usbip attach --remote="${EXECUTION_HOST_IP}" --busid=4-2
# sudo usbip attach --remote="${EXECUTION_HOST_IP}" --busid=4-3
# sudo usbip attach --remote="${EXECUTION_HOST_IP}" --busid=4-4

# COM6 / left follower 的 BUSID：
sudo usbip attach --remote="${EXECUTION_HOST_IP}" --busid=TODO_COM6_BUSID

# COM8 / right follower 的 BUSID：
sudo usbip attach --remote="${EXECUTION_HOST_IP}" --busid=TODO_COM8_BUSID

# COM4 / left leader 的 BUSID：
sudo usbip attach --remote="${EXECUTION_HOST_IP}" --busid=TODO_COM4_BUSID

# COM9 / right leader 的 BUSID：
sudo usbip attach --remote="${EXECUTION_HOST_IP}" --busid=TODO_COM9_BUSID

#實際連線時解完問題後的用法
sudo /usr/lib/linux-tools/5.15.0-177-generic/usbip attach --remote=192.168.50.161 --busid=1-10
#相機可使用相同方式連接

# 6. 檢查 WSL 是否看到 USB 和 serial device。
lsusb
dmesg | tail -80
ls -l /dev/serial/by-id/ 2>/dev/null || true
ls -l /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || true

# 7. 如果權限不足，先暫時用 sudo 測試；之後再加 udev 規則或把 user 加到 dialout。
groups
