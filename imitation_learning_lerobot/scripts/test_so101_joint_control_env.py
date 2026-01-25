"""
SO101JointControlEnv 測試腳本
驗證環境載入、觀測格式和基本功能
"""
from imitation_learning_lerobot.envs import SO101JointControlEnv
import numpy as np

def test_env():
    print("=" * 50)
    print("SO101JointControlEnv 功能測試")
    print("=" * 50)
    
    # 1. 測試環境初始化
    print("\n[TEST 1] 環境初始化...")
    env = SO101JointControlEnv()
    print(f"  ✓ 環境名稱: {env.name}")
    print(f"  ✓ Action 維度: {env.action_dim}")
    print(f"  ✓ State 維度: {env.state_dim}")
    print(f"  ✓ 控制頻率: {env.control_hz} Hz")
    
    # 2. 測試 reset
    print("\n[TEST 2] 環境 Reset...")
    obs, info = env.reset()
    print(f"  ✓ 觀測 keys: {list(obs.keys())}")
    print(f"  ✓ observation.state shape: {obs['observation.state'].shape}")
    print(f"  ✓ observation.images.camera1 shape: {obs['observation.images.camera1'].shape}")
    print(f"  ✓ observation.images.camera3 shape: {obs['observation.images.camera3'].shape}")
    
    # 3. 測試 step
    print("\n[TEST 3] 環境 Step (關節角度控制)...")
    # 輸入 6 維關節角度（度）
    action = np.array([10.0, 20.0, 30.0, 0.0, 0.0, 45.0])  # shoulder_pan, shoulder_lift, etc.
    print(f"  輸入 action (度): {action}")
    
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  ✓ Step 完成")
    print(f"  ✓ 新 state: {obs['observation.state']}")
    
    # 4. 執行幾步驗證運動
    print("\n[TEST 4] 連續執行 10 步...")
    for i in range(10):
        action = np.array([10.0 * (i+1), 0.0, 0.0, 0.0, 0.0, 0.0])
        obs, _, _, _, _ = env.step(action)
    print(f"  ✓ 10 步執行完成")
    print(f"  ✓ 最終 state: {obs['observation.state']}")
    
    # 5. 關閉環境
    env.close()
    print("\n" + "=" * 50)
    print("所有測試通過！")
    print("=" * 50)

if __name__ == "__main__":
    test_env()
