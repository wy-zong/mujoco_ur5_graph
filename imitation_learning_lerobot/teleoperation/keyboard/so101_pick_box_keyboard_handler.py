"""
SO101 Pick Box Keyboard Handler
Keyboard teleoperation handler for SO101-only environment.
Produces 7-dimensional action vector.
"""
import numpy as np
from pynput import keyboard

from ..handler import Handler


class SO101PickBoxKeyboardHandler(Handler):
    _name = "so101_pick_box_keyboard"

    def __init__(self):
        super().__init__()

        # 7 維 action: dr, dtheta, dz, droll, dpitch, gripper, reserved
        self._action = np.zeros(7)

        self._done = False
        self._listener: keyboard.Listener = None

    def on_press(self, key):
        try:
            key_char = key.char
        except AttributeError:
            key_char = str(key)

        if not self._sync:
            if key_char == 'Key.ctrl_r':
                self._sync = True
            return
        else:
            if key_char == 'Key.shift_r':
                self._sync = False
                return

        if key_char == 'Key.enter':
            self._done = True
            return

        # ========== SO101 控制 ==========
        # 數字鍵盤和特殊鍵皆可使用

        # Radius (dr): 1/7 or End/Home
        if key_char == '1' or key_char == 'Key.end':
            self._action[0] = +1.0
        if key_char == '7' or key_char == 'Key.home':
            self._action[0] = -1.0

        # Theta (dtheta): 4/6 or Left/Right
        if key_char == '4' or key_char == 'Key.left':
            self._action[1] = +1.0
        if key_char == '6' or key_char == 'Key.right':
            self._action[1] = -1.0

        # Z (dz): 8/2 or Up/Down
        if key_char == '8' or key_char == 'Key.up':
            self._action[2] = +1.0
        if key_char == '2' or key_char == 'Key.down':
            self._action[2] = -1.0

        # Roll: / *
        if key_char == '/':
            self._action[3] = +1.0
        if key_char == '*':
            self._action[3] = -1.0

        # Pitch: - +
        if key_char == '-':
            self._action[4] = +1.0
        if key_char == '+':
            self._action[4] = -1.0

        # Gripper: 9/3 or PageUp/PageDown (按住模式)
        # 按住 9/PageUp = 關閉方向 (+1)
        # 按住 3/PageDown = 打開方向 (-1)
        if key_char == '9' or key_char == 'Key.page_up':
            self._action[5] = +1.0  # 關閉方向
        if key_char == '3' or key_char == 'Key.page_down':
            self._action[5] = -1.0  # 打開方向

    def on_release(self, key):
        try:
            key_char = key.char
        except AttributeError:
            key_char = str(key)

        # 放開鍵清零對應方向
        if key_char in ['1', '7', 'Key.end', 'Key.home']:
            self._action[0] = 0.0
        if key_char in ['6', '4', 'Key.left', 'Key.right']:
            self._action[1] = 0.0
        if key_char in ['8', '2', 'Key.up', 'Key.down']:
            self._action[2] = 0.0
        if key_char in ['/', '*']:
            self._action[3] = 0.0
        if key_char in ['-', '+']:
            self._action[4] = 0.0
        # 夾爪：放開時停止移動
        if key_char in ['9', '3', 'Key.page_up', 'Key.page_down']:
            self._action[5] = 0.0

    def start(self):
        self._listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self._listener.start()

    def close(self):
        self._listener.stop()

    def print_info(self):
        print("=" * 50)
        print("SO101 Pick Box Keyboard Handler")
        print("=" * 50)
        print("Start:           Right Ctrl")
        print("Pause:           Right Shift")
        print("Stop:            Enter")
        print("-" * 50)
        print("Extend (dr+):    Keypad 1 / End")
        print("Retract (dr-):   Keypad 7 / Home")
        print("Rotate Left:     Keypad 4 / Left Arrow")
        print("Rotate Right:    Keypad 6 / Right Arrow")
        print("Up (dz+):        Keypad 8 / Up Arrow")
        print("Down (dz-):      Keypad 2 / Down Arrow")
        print("Roll+:           /")
        print("Roll-:           *")
        print("Pitch+:          -")
        print("Pitch-:          +")
        print("Gripper Close:   Keypad 9 / Page Up")
        print("Gripper Open:    Keypad 3 / Page Down")
        print("=" * 50)


# Action mapping (7 維):
#   0: dr_cmd     - 半徑增量 (前後伸縮)
#   1: dtheta_cmd - 角度增量 (底座旋轉)
#   2: dz         - Z 軸增量 (上下)
#   3: droll      - 夾爪滾轉
#   4: dpitch     - 手腕俯仰
#   5: gripper    - 夾爪開合 (0-1)
#   6: reserved   - 保留
