# detach 分兩種情況：
#
# 1. 如果是推理主機 WSL 用 Linux usbip attach：
#    在推理主機 WSL 裡查 port，再 detach。
#
# 2. 如果是本機 Windows 用 usbipd attach --wsl：
#    在該 Windows PowerShell 裡 usbipd detach --busid <BUSID>。

# 推理主機 WSL：
usbip port

# 依 usbip port 顯示的 port number detach，例如：
# sudo usbip detach --port=00
# sudo usbip detach --port=01
# sudo usbip detach --port=02
# sudo usbip detach --port=03

# 執行主機 Windows 若要解除分享，可用系統管理員 PowerShell：
# usbipd unbind --busid <BUSID>
