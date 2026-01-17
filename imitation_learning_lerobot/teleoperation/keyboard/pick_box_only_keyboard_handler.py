# pick_box_only_keyboard_handler.py
"""
Keyboard handler for PickBoxOnlyEnv - 繼承自 PickBoxKeyboardHandler
"""
from .pick_box_keyboard_handler import PickBoxKeyboardHandler


class PickBoxOnlyKeyboardHandler(PickBoxKeyboardHandler):
    """PickBoxOnlyEnv 的鍵盤控制 handler，繼承 PickBoxKeyboardHandler 的所有功能"""
    _name = "pick_box_only_keyboard"

    def __init__(self):
        super().__init__()
        print("[PickBoxOnlyKeyboardHandler] 使用 pick_box_only 環境控制（純物理夾取模式）")
