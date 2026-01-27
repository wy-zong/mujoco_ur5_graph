from typing import Type
from pathlib import Path
import argparse
import sys

# Add project root to path for cross-platform compatibility
_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from loop_rate_limiters import RateLimiter
import numpy as np
import h5py
import cv2

from imitation_learning_lerobot.envs import Env, EnvFactory
from imitation_learning_lerobot.teleoperation import HandlerFactory


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--env.type',
        type=str,
        dest='env_type',
        required=True,
        help='env type'
    )

    parser.add_argument(
        '--handler.type',
        type=str,
        dest='handler_type',
        required=True,
        help='handler type'
    )

    return parser.parse_args()


def teleoperate(env_cls: Type[Env], handler_type):
    """
    執行遙操作資料收集流程
    
    Args:
        env_cls: 環境類別，包含環境配置資訊
        handler_type: 操作處理器類型（例如 'keyboard'）
    
    Returns:
        dict: 包含觀察（observations）和動作（actions）的資料字典
              - '/observations/agent_pos': 機器人關節位置列表
              - '/observations/pixels/{camera}': 各相機的影像列表  
              - '/actions': 執行的動作列表
    """
    handler_cls = HandlerFactory.get_strategies(env_cls.name + "_" + handler_type)
    handler = handler_cls()
    handler.start()
    handler.print_info()

    env = env_cls(render_mode="human")
    observation, info = env.reset()

    for camera in env_cls.cameras:
        cv2.namedWindow(camera, cv2.WINDOW_GUI_NORMAL)

    data_dict = {
        '/observations/agent_pos': [],
        **{f'/observations/pixels/{camera}': [] for camera in env_cls.cameras},
        '/actions': []
    }

    rate_limiter = RateLimiter(frequency=env.control_hz)

    action = handler.action
    while not handler.done:
        if not handler.sync:
            rate_limiter.sleep()
            continue

        action[:] = handler.action
        
        # 記錄當前觀察和動作
        # 注意：這裡記錄的是 observation_t 和即將執行的 action_t
        data_dict['/observations/agent_pos'].append(observation['agent_pos'])
        for camera in env_cls.cameras:
            data_dict[f'/observations/pixels/{camera}'].append(observation['pixels'][camera])
        data_dict['/actions'].append(action.copy())
        
        # [已移除] 動作幅度過濾功能
        # 原本的設計會跳過變化小於閾值的動作，但這會導致以下問題：
        # 1. 破壞時間序列的連續性，影響模仿學習的訓練
        # 2. 丟失重要的靜止狀態資訊（模型需要學習何時該停止）
        # 3. 造成 observation-action 配對不一致
        # 因此改為記錄所有 frames，保留完整的軌跡資料

        observation, reward, terminated, truncated, info = env.step(action)

        env.render()
        for camera in env.cameras:
            cv2.imshow(camera, cv2.cvtColor(observation["pixels"][camera], cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

        rate_limiter.sleep()

    cv2.destroyAllWindows()
    handler.close()
    env.close()

    return data_dict


def write_to_h5(env_cls: Type[Env], data_dict: dict):
    """
    將收集的資料寫入 HDF5 檔案
    
    Args:
        env_cls: 環境類別，用於獲取資料維度和相機配置
        data_dict: 包含觀察和動作的資料字典
    
    檔案結構：
        outputs/datasets/{env_name}_hdf5/episode_{index:06d}.hdf5
        ├── observations/
        │   ├── agent_pos: (T, state_dim) 關節位置
        │   └── pixels/
        │       └── {camera}: (T, H, W, 3) 影像資料
        └── actions: (T, action_dim) 動作序列
    """
    h5_dir = Path(__file__).parent.parent.parent / Path("outputs/datasets") / Path(env_cls.name + "_hdf5")
    h5_dir.mkdir(parents=True, exist_ok=True)

    index = len([f for f in h5_dir.iterdir() if f.is_file()])

    h5_path = h5_dir / Path(f"episode_{index:06d}.hdf5")

    with h5py.File(h5_path, 'w', ) as root:

        episode_length = len(data_dict['/actions'])

        obs = root.create_group('observations')

        obs.create_dataset('agent_pos', (episode_length, env_cls.state_dim), dtype='float32', compression='gzip')

        pixels = obs.create_group('pixels')
        for camera in env_cls.cameras:
            shape = (episode_length, env_cls.height, env_cls.width, 3)
            chunks = (1, env_cls.height, env_cls.width, 3)
            pixels.create_dataset(camera, shape=shape, dtype='uint8', chunks=chunks, compression='gzip')

        root.create_dataset('actions', (episode_length, env_cls.action_dim), dtype='float32', compression='gzip')

        for name, array in data_dict.items():
            root[name][...] = array
    
    print(f"✓ 資料已儲存至: {h5_path}")
    print(f"✓ Episode 長度: {episode_length} frames")


from imitation_learning_lerobot.envs.scripted_flow import scripted_pick_and_place

def main():
    """
    資料收集主程式
    
    支援兩種模式：
    1. 腳本化自動收集：--handler.type=script/scripted/auto
    2. 手動遙操作收集：--handler.type=keyboard 等
    """
    args = parse_args()
    env_cls = EnvFactory.get_strategies(args.env_type)

    # 腳本化自動收集模式
    if args.handler_type.lower() in ["script", "scripted", "auto"]:
        scripted_pick_and_place(env_cls)
        return

    # 手動/鍵盤遙操作路徑
    data_dict = teleoperate(env_cls, args.handler_type)
    
    # 資料驗證：確保有收集到資料才進行儲存
    if not data_dict['/actions']:
        print("⚠ 警告：沒有收集到任何資料，取消儲存")
        return
    
    print(f"\n📊 收集統計：")
    print(f"   - 總 frames: {len(data_dict['/actions'])}")
    print(f"   - 相機數量: {len(env_cls.cameras)}")
    
    write_to_h5(env_cls, data_dict)

if __name__ == '__main__':
    main()
