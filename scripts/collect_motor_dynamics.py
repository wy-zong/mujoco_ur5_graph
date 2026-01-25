#!/usr/bin/env python
"""
Motor Dynamics Data Collection Script - 馬達動態數據收集腳本

用於收集實機馬達的動態響應數據，輸出 CSV 檔案供 MuJoCo 模擬校準使用。

收集的數據：
1. Goal_Position vs Present_Position（追蹤誤差）
2. Present_Velocity（實際速度）
3. Present_Load（負載/扭矩）
4. Present_Current（電流）
5. PID 參數

Usage:
    python scripts/collect_motor_dynamics.py --port COM5 --id so101_follower_arm

輸出檔案：
    scripts/motor_dynamics_<timestamp>.csv
"""
import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig


def read_motor_params(bus):
    """讀取馬達的 PID 參數（只需讀取一次）"""
    params = {}
    for motor in bus.motors:
        params[motor] = {
            'P_Coefficient': bus.read('P_Coefficient', motor),
            'D_Coefficient': bus.read('D_Coefficient', motor),
            'I_Coefficient': bus.read('I_Coefficient', motor),
        }
    return params


def read_motor_dynamics(bus):
    """讀取馬達當前動態狀態"""
    data = {}
    
    # 同步讀取位置（最重要）
    positions = bus.sync_read('Present_Position')
    for motor, pos in positions.items():
        data[f'{motor}_pos'] = pos
    
    # 讀取速度
    try:
        velocities = bus.sync_read('Present_Velocity')
        for motor, vel in velocities.items():
            data[f'{motor}_vel'] = vel
    except Exception as e:
        print(f"[WARN] Cannot read velocity: {e}")
    
    # 讀取負載（扭矩）
    try:
        loads = bus.sync_read('Present_Load')
        for motor, load in loads.items():
            data[f'{motor}_load'] = load
    except Exception as e:
        print(f"[WARN] Cannot read load: {e}")
    
    # 讀取電流
    try:
        currents = bus.sync_read('Present_Current')
        for motor, curr in currents.items():
            data[f'{motor}_current'] = curr
    except Exception as e:
        print(f"[WARN] Cannot read current: {e}")
    
    return data


def collect_tracking_data(
    leader: SO101Leader,
    follower: SO101Follower,
    duration_sec: float = 30.0,
    fps: int = 30
):
    """
    收集追蹤數據：Leader 發出命令，Follower 執行，記錄差異
    
    這是最關鍵的數據：
    - goal_pos: Leader 發出的目標位置
    - present_pos: Follower 的實際位置
    - tracking_error = goal_pos - present_pos
    """
    output_file = Path(__file__).parent / f"motor_dynamics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    print(f"\n收集追蹤數據，持續 {duration_sec} 秒...")
    print(f"輸出檔案: {output_file}")
    print("請移動 Leader 手臂！\n")
    
    # 讀取 PID 參數（只需一次）
    pid_params = read_motor_params(follower.bus)
    print("馬達 PID 參數:")
    for motor, params in pid_params.items():
        print(f"  {motor}: P={params['P_Coefficient']}, I={params['I_Coefficient']}, D={params['D_Coefficient']}")
    print()
    
    # 準備 CSV 欄位
    motor_names = list(follower.bus.motors.keys())
    fieldnames = ['timestamp', 'step']
    for motor in motor_names:
        fieldnames.extend([
            f'{motor}_goal_pos',
            f'{motor}_present_pos',
            f'{motor}_tracking_error',
            f'{motor}_velocity',
            f'{motor}_load',
        ])
    
    data_rows = []
    step = 0
    start_time = time.perf_counter()
    
    try:
        while time.perf_counter() - start_time < duration_sec:
            loop_start = time.perf_counter()
            
            # 1. 讀取 Leader 目標位置
            goal_action = leader.get_action()
            
            # 2. 發送到 Follower
            follower.send_action(goal_action)
            
            # 3. 讀取 Follower 實際狀態
            dynamics = read_motor_dynamics(follower.bus)
            
            # 4. 記錄數據
            row = {
                'timestamp': time.perf_counter() - start_time,
                'step': step
            }
            
            for motor in motor_names:
                goal_key = f'{motor}.pos'
                goal_pos = goal_action.get(goal_key, 0.0)
                present_pos = dynamics.get(f'{motor}_pos', 0.0)
                
                row[f'{motor}_goal_pos'] = goal_pos
                row[f'{motor}_present_pos'] = present_pos
                row[f'{motor}_tracking_error'] = goal_pos - present_pos
                row[f'{motor}_velocity'] = dynamics.get(f'{motor}_vel', 0.0)
                row[f'{motor}_load'] = dynamics.get(f'{motor}_load', 0.0)
            
            data_rows.append(row)
            step += 1
            
            # 控制頻率
            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0, 1.0 / fps - elapsed)
            time.sleep(sleep_time)
            
            if step % 30 == 0:
                print(f"已收集 {step} 筆資料 ({time.perf_counter() - start_time:.1f}s)")
                
    except KeyboardInterrupt:
        print("\n提前停止收集")
    
    # 寫入 CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_rows)
    
    print(f"\n完成！共收集 {len(data_rows)} 筆資料")
    print(f"輸出檔案: {output_file}")
    
    # 計算追蹤誤差統計
    print("\n追蹤誤差統計 (degrees):")
    for motor in motor_names:
        errors = [abs(row[f'{motor}_tracking_error']) for row in data_rows]
        if errors:
            avg_error = sum(errors) / len(errors)
            max_error = max(errors)
            print(f"  {motor}: 平均誤差={avg_error:.2f}°, 最大誤差={max_error:.2f}°")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Collect motor dynamics data for MuJoCo calibration")
    parser.add_argument('--leader_port', type=str, default='COM6', help='Leader arm port')
    parser.add_argument('--follower_port', type=str, default='COM5', help='Follower arm port')
    parser.add_argument('--leader_id', type=str, default='so101_leader_arm')
    parser.add_argument('--follower_id', type=str, default='so101_follower_arm')
    parser.add_argument('--duration', type=float, default=30.0, help='Collection duration (seconds)')
    parser.add_argument('--fps', type=int, default=30, help='Sampling rate')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Motor Dynamics Data Collection")
    print("=" * 60)
    print(f"Leader port:   {args.leader_port}")
    print(f"Follower port: {args.follower_port}")
    print(f"Duration:      {args.duration}s")
    print(f"FPS:           {args.fps}")
    print("=" * 60)
    
    # 初始化
    leader_config = SO101LeaderConfig(
        port=args.leader_port, id=args.leader_id, use_degrees=True
    )
    leader = SO101Leader(leader_config)
    
    follower_config = SO101FollowerConfig(
        port=args.follower_port, id=args.follower_id, use_degrees=True
    )
    follower = SO101Follower(follower_config)
    
    # 連接
    print("\n連接裝置...")
    leader.connect()
    print(f"  ✓ Leader connected ({args.leader_port})")
    follower.connect()
    print(f"  ✓ Follower connected ({args.follower_port})")
    
    try:
        output_file = collect_tracking_data(
            leader, follower,
            duration_sec=args.duration,
            fps=args.fps
        )
        
        print("\n" + "=" * 60)
        print("數據收集完成！")
        print(f"請將 {output_file} 分享給我進行分析")
        print("=" * 60)
        
    finally:
        print("\n中斷連接...")
        leader.disconnect()
        follower.disconnect()
        print("完成")


if __name__ == "__main__":
    main()
