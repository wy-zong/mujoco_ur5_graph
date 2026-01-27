from typing import Type
from pathlib import Path
import argparse
import dataclasses
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from imitation_learning_lerobot.envs import Env, EnvFactory


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


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
        '--episode',
        type=int,
        default=100,
        help='episode'
    )

    parser.add_argument(
        '--display',
        action='store_true',
        default=False,
        help='顯示 MuJoCo 視覺化介面 (預設: 不顯示)'
    )

    return parser.parse_args()


def create_empty_dataset(env_cls: Type[Env]):
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(env_cls.states),),
            "names": {
                "position": env_cls.states,
            }
        }, "action": {
            "dtype": "float32",
            "shape": (len(env_cls.states),),
            "names": {
                "position": env_cls.states,
            }
        }
    }

    for camera in env_cls.cameras:
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": (env_cls.height, env_cls.width, 3),
            "names": [
                "height",
                "width",
                "channel"
            ]
        }

    config = DatasetConfig()

    dataset = LeRobotDataset.create(
        repo_id=env_cls.name,
        fps=env_cls.control_hz,
        features=features,
        root=Path(__file__).parent.parent.parent / Path("outputs/datasets") / Path(env_cls.name),
        robot_type=env_cls.robot_type,
        use_videos=config.use_videos,
        tolerance_s=config.tolerance_s,
        image_writer_processes=config.image_writer_processes,
        image_writer_threads=config.image_writer_threads,
        video_backend=config.video_backend
    )

    return dataset


def populate_dataset(episode: int, env_cls: Type[Env], dataset: LeRobotDataset, display: bool = False):
    render_mode = "human" if display else "rgb_array"
    env = env_cls(render_mode=render_mode)
    task = env.name
    
    successful_episodes = 0
    failed_episodes = 0
    
    for i in range(episode):
        try:
            data = env.run(keep_state=(i>0))
            
            # 檢查成功標記 (如果環境有實作)
            is_success = data.get("success", True)
            
            if not is_success:
                print(f"[WARN] Episode {i+1} 夾取失敗,但仍儲存資料")
                failed_episodes += 1
            else:
                successful_episodes += 1
            
            episode_length = len(data["observations"])
            
            for j in range(episode_length):
                frame = {
                    "observation.state": data["observations"][j]["agent_pos"],
                    "action": data["actions"][j],
                    "task": task,  # 新版 API: task 放入 frame 字典中
                }
                
                for camera in env_cls.cameras:
                    frame[f"observation.images.{camera}"] = data["observations"][j]["pixels"][camera]
                
                dataset.add_frame(frame)
            
            dataset.save_episode()
            print(f"Episode {i+1}/{episode} 完成 (成功: {successful_episodes}, 失敗: {failed_episodes})")
            
        except Exception as e:
            print(f"[ERROR] Episode {i+1} 執行失敗: {e}")
            import traceback
            traceback.print_exc()
            failed_episodes += 1
            
            # 清空緩衝區,跳過此 episode
            if hasattr(dataset, 'clear_episode_buffer'):
                dataset.clear_episode_buffer()
            continue
    
    env.close()
    
    print(f"\n最終統計: 成功 {successful_episodes}/{episode}, 失敗 {failed_episodes}/{episode}")



def main():
    args = parse_args()

    env_type = args.env_type
    env_cls = EnvFactory.get_strategies(env_type)

    dataset = create_empty_dataset(env_cls)
    
    # 檢查是否已有資料
    existing_episodes = dataset.num_episodes
    if existing_episodes > 0:
        print(f"\n檢測到已存在 {existing_episodes} 個 episodes")
        response = input(f"是否繼續蒐集至 {args.episode} 個 episodes? (y/n): ").strip().lower()
        if response != 'y':
            print("已取消")
            return
        print(f"將繼續蒐集...")
    
    # 計算實際需要蒐集的數量
    episodes_to_collect = max(0, args.episode - existing_episodes)
    
    if episodes_to_collect > 0:
        populate_dataset(episodes_to_collect, env_cls, dataset, display=args.display)
    else:
        print(f"已達到目標 episodes 數量 ({args.episode}),無需繼續蒐集")
    
    # 輸出資料集統計資訊
    print("\n" + "="*60)
    print("資料集蒐集完成!")
    print("="*60)
    print(f"環境類型: {env_type}")
    print(f"目標 episodes: {args.episode}")
    print(f"實際 episodes: {dataset.num_episodes}")
    print(f"總幀數: {dataset.num_frames}")
    print(f"平均每 episode 幀數: {dataset.num_frames / max(dataset.num_episodes, 1):.1f}")
    print(f"資料集路徑: {dataset.root}")
    print(f"控制頻率: {dataset.fps} Hz")
    print(f"相機: {', '.join(env_cls.cameras)}")
    print("="*60)


if __name__ == '__main__':
    main()
