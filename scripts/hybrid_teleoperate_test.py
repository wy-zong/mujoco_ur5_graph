#!/usr/bin/env python
"""
Hybrid Teleoperation Script - Test Version
混合遙操作腳本：Leader 手臂同時控制 Follower 和 MuJoCo 模擬環境

Usage:
    python scripts/hybrid_teleoperate_test.py --leader_port=COM6 --follower_port=COM5
    
功能：
- 讀取 leader 手臂的關節位置
- 發送到 follower 手臂（現有行為）
- 直接映射到模擬環境（新功能）
"""
import argparse
import time
import logging
from pprint import pformat

# LeRobot imports
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

# MuJoCo simulation environment (hybrid version)
from imitation_learning_lerobot.envs.so101_pick_box_env_hybrid import SO101PickBoxEnvHybrid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def hybrid_teleop_loop(
    leader: SO101Leader,
    follower: SO101Follower,
    sim_env: SO101PickBoxEnvHybrid,
    fps: int = 30,
    display_data: bool = True,
):
    """
    雙軌遙操作控制迴圈
    
    Args:
        leader: SO101 leader 手臂（遙操作器）
        follower: SO101 follower 手臂（被控制）
        sim_env: MuJoCo 模擬環境 (test 版本)
        fps: 控制頻率
        display_data: 是否顯示資料
    """
    print("\n" + "=" * 60)
    print("Hybrid Teleoperation Started")
    print("=" * 60)
    print("Leader  → Follower (Physical)")
    print("Leader  → MuJoCo Simulation")
    print("-" * 60)
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")
    
    step_num = 0
    
    try:
        while True:
            loop_start = time.perf_counter()
            
            # 1. 讀取 leader 關節位置
            # 格式: {'shoulder_pan.pos': deg, 'shoulder_lift.pos': deg, ...}
            action = leader.get_action()
            
            # 2. 發送到 follower 手臂（現有行為）
            sent_action = follower.send_action(action)
            
            # 3. 直接映射到模擬環境
            # set_joint_positions 接受與 leader.get_action() 相同格式
            sim_obs = sim_env.set_joint_positions(action)
            sim_env.render()
            
            # 顯示資料
            if display_data and step_num % 10 == 0:
                print(f"\n[Step {step_num}]")
                print("Joint Positions (deg):")
                for key, val in action.items():
                    print(f"  {key}: {val:.2f}")
            
            step_num += 1
            
            # 控制頻率
            dt_s = time.perf_counter() - loop_start
            sleep_time = max(0, 1 / fps - dt_s)
            time.sleep(sleep_time)
            
            loop_time = time.perf_counter() - loop_start
            if step_num % 30 == 0:
                print(f"[PERF] Loop: {loop_time*1000:.1f}ms ({1/loop_time:.0f} Hz)")
                
    except KeyboardInterrupt:
        print("\n\nStopping hybrid teleoperation...")


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Teleoperation: Control both physical follower and MuJoCo simulation"
    )
    parser.add_argument(
        "--leader_port",
        type=str,
        default="COM6",
        help="Leader arm serial port (default: COM6)"
    )
    parser.add_argument(
        "--follower_port",
        type=str,
        default="COM5",
        help="Follower arm serial port (default: COM5)"
    )
    parser.add_argument(
        "--leader_id",
        type=str,
        default="so101_leader_arm",
        help="Leader arm ID for calibration"
    )
    parser.add_argument(
        "--follower_id",
        type=str,
        default="so101_follower_arm",
        help="Follower arm ID for calibration"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Control frequency in Hz (default: 30)"
    )
    parser.add_argument(
        "--sim_only",
        action="store_true",
        help="Only run simulation (no physical follower)"
    )
    parser.add_argument(
        "--display_data",
        type=bool,
        default=True,
        help="Display joint data"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("Initializing Hybrid Teleoperation (Test Version)")
    print("=" * 60)
    print(f"Leader port:   {args.leader_port}")
    print(f"Follower port: {args.follower_port}")
    print(f"FPS:           {args.fps}")
    print(f"Sim only:      {args.sim_only}")
    print("=" * 60 + "\n")
    
    # 初始化 leader
    leader_config = SO101LeaderConfig(
        port=args.leader_port,
        id=args.leader_id,
        use_degrees=True,  # 使用度數
    )
    leader = SO101Leader(leader_config)
    
    # 初始化 follower (如果不是 sim_only 模式)
    follower = None
    if not args.sim_only:
        follower_config = SO101FollowerConfig(
            port=args.follower_port,
            id=args.follower_id,
            use_degrees=True,
        )
        follower = SO101Follower(follower_config)
    
    # 初始化模擬環境 (hybrid 版本)
    sim_env = SO101PickBoxEnvHybrid(render_mode="human")
    
    # 連接裝置
    print("Connecting devices...")
    leader.connect()
    print(f"  ✓ Leader connected ({args.leader_port})")
    
    if follower is not None:
        follower.connect()
        print(f"  ✓ Follower connected ({args.follower_port})")
    
    # 重置模擬環境
    print("Resetting simulation environment...")
    sim_env.reset()
    print("  ✓ Simulation ready")
    
    try:
        if args.sim_only:
            # 只控制模擬環境
            sim_only_loop(leader, sim_env, args.fps, args.display_data)
        else:
            # 雙軌控制
            hybrid_teleop_loop(
                leader=leader,
                follower=follower,
                sim_env=sim_env,
                fps=args.fps,
                display_data=args.display_data,
            )
    finally:
        print("\nDisconnecting devices...")
        leader.disconnect()
        print("  ✓ Leader disconnected")
        
        if follower is not None:
            follower.disconnect()
            print("  ✓ Follower disconnected")
        
        sim_env.close()
        print("  ✓ Simulation closed")
        
        print("\nHybrid teleoperation ended.")


def sim_only_loop(
    leader: SO101Leader,
    sim_env: SO101PickBoxEnvHybrid,
    fps: int = 30,
    display_data: bool = True,
):
    """
    只控制模擬環境（用於測試）
    """
    print("\n" + "=" * 60)
    print("Simulation-Only Mode")
    print("=" * 60)
    print("Leader  → MuJoCo Simulation")
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")
    
    step_num = 0
    
    try:
        while True:
            loop_start = time.perf_counter()
            
            # 讀取 leader 關節位置
            action = leader.get_action()
            
            # 直接映射到模擬環境
            sim_obs = sim_env.set_joint_positions(action)
            sim_env.render()
            
            if display_data and step_num % 10 == 0:
                print(f"\n[Step {step_num}]")
                for key, val in action.items():
                    print(f"  {key}: {val:.2f}")
            
            step_num += 1
            
            dt_s = time.perf_counter() - loop_start
            sleep_time = max(0, 1 / fps - dt_s)
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\n\nStopping simulation...")


if __name__ == "__main__":
    main()
