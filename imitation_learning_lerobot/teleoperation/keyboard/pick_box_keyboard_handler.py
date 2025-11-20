import numpy as np
from pynput import keyboard

from ..handler import Handler


class PickBoxKeyboardHandler(Handler):
    _name = "pick_box_keyboard"

    def __init__(self):
        super().__init__()

        # self._action = np.zeros(4) # 單手臂 ur5
        self._action = np.zeros(14) # 雙手臂 ur5+so101

        self._done = False
        self._vel = 0.005

        self._listener: keyboard.Listener = None

        # 哪一隻手臂在控制：0=UR5, 1=SO101
        self._active_arm = 0
        

    def on_press(self, key):
        try:
            key_char = key.char
        except AttributeError:
            key_char = str(key)

        if not self._sync:

            if key_char == 'Key.ctrl_r':
                self._sync = True
        else:
            if key_char == 'Key.shift_r':
                self._sync = False

        if key_char == 'Key.enter':
            self._done = True

        if not self._sync:
            return
        
        # ===== 新增：切換控制哪一隻手臂 =====
        if key_char in ['0', '.']:
            self._active_arm = 1 - self._active_arm
            print(f"[INFO] Active arm: {'UR5' if self._active_arm == 0 else 'SO101'}")
            return

        # ========== UR5 控制（跟原本一樣，只是掛在 active_arm==0）==========
        if self._active_arm == 0:
            if key_char == '2':
                self._action[2] += self._vel

            if key_char == '8':
                self._action[2] -= self._vel

            if key_char == "6":
                self._action[0] -= self._vel

            if key_char == '4':
                self._action[0] += self._vel

            if key_char == '7':
                self._action[1] += self._vel

            if key_char == '1':
                self._action[1] -= self._vel

            if key_char == '9':
                self._action[3] += 0.05

            if key_char == '3':
                self._action[3] -= 0.05

            if key_char == '/':
                self._action[4] += 0.05

            if key_char == '*':
                self._action[4] -= 0.05

            if key_char == '-':
                self._action[5] += 0.05
            
            if key_char == '+':
                self._action[5] -= 0.05

            if key_char == '5':
                self._action[6] += 0.05

            if key_char == '0':
                self._action[6] -= 0.05

            self._action[3] = np.clip(self._action[3], 0.0, 1.0)

        # ========== SO101 控制（六個 joint 的 "速度指令"）==========
        else:
            so_vel = 0.005  # 每按一次鍵，對應 action ±1

            
            if key_char == '2':
                self._action[9] += so_vel
            if key_char == '8':
                self._action[9] -= so_vel 

            
            if key_char == '4':
                self._action[7] += so_vel
            if key_char == '6':
                self._action[7] -= so_vel

            
            if key_char == '1':
                self._action[8] += so_vel
            if key_char == '7':
                self._action[8] -= so_vel

            
            if key_char == '9':
                self._action[10] += so_vel
            if key_char == '3':
                self._action[10] -= so_vel

            
            if key_char == '/':
                self._action[11] += so_vel
            if key_char == '*':
                self._action[11] -= so_vel

            
            if key_char == '-':
                self._action[12] += so_vel
            if key_char == '+':
                self._action[12] -= so_vel

               
            if key_char == '5':
                self._action[13] += so_vel
            if key_char == '.':
                self._action[13] -= so_vel


            
            self._action[10] = np.clip(self._action[10], 0.0, 1.0)

    def start(self):
        self._listener = keyboard.Listener(on_press=self.on_press)
        self._listener.start()

    def close(self):
        self._listener.stop()

    def print_info(self):
        print("------------------------------")
        print("Start:           Right Ctrl")
        print("Pause:           Right Shift")
        print("Stop:            Enter")
        print("+X:              Keypad 1")
        print("-X:              Keypad 7")
        print("+Y:              Keypad 6")
        print("-Y:              Keypad 4")
        print("+Z:              Keypad 8")
        print("-Z:              Keypad 2")
        print("Open:            Keypad 3")
        print("Close:           Keypad 9")

    


# action: 長度 14

# UR5:
#   0: ur5_dx
#   1: ur5_dy
#   2: ur5_dz
#   3: ur5_gripper (0~1)
#   4: ur5_roll
#   5: ur5_pitch
#   6: ur5_yaw

# SO101:
#   7: so_dx
#   8: so_dy
#   9: so_dz
#  10: so_gripper (0~1)
#  11: so_roll
#  12: so_pitch
#  13: so_yaw