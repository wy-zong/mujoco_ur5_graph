import sys
import os
import glob
import json
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import numpy as np
import pandas as pd

def analyze_dataset(dataset_path):
    info_path = os.path.join(dataset_path, 'meta', 'info.json')
    if not os.path.exists(info_path):
        print(f"Error: Could not find meta/info.json in {dataset_path}")
        return

    with open(info_path, 'r') as f:
        info = json.load(f)
    
    fps = info.get('fps', 30)
    data_files = sorted(glob.glob(os.path.join(dataset_path, 'data', '*/*.parquet')))
    
    if not data_files:
        print(f"Error: No parquet files found in {os.path.join(dataset_path, 'data')}")
        return

    print(f"=== Analyzing Dataset: {dataset_path} ===")
    
    # 1. Info & Tasks
    print("\n--- 1. General Info & Tasks ---")
    print(f"Codebase Version: {info.get('codebase_version', 'N/A')}")
    print(f"Robot Type: {info.get('robot_type', 'N/A')}")
    print(f"Total Episodes (Info): {info.get('total_episodes', 'N/A')}")
    print(f"Total Frames (Info): {info.get('total_frames', 'N/A')}")
    print(f"FPS: {fps}")
    
    tasks_file = os.path.join(dataset_path, 'meta', 'tasks.parquet')
    if os.path.exists(tasks_file):
        try:
            tasks_df = pq.read_table(tasks_file).to_pandas()
            print("\nTasks Details:")
            for index, row in tasks_df.iterrows():
                print(f"  - '{index}' (task_index: {row.get('task_index', 'N/A')})")
        except Exception as e:
            print(f"Could not read tasks.parquet: {e}")

    # 2. Rule Scores Evaluation (if exists)
    print("\n--- 2. Rule Scores (Evaluation Metrics) ---")
    rule_file = os.path.join(dataset_path, 'rule_scores.parquet')
    if os.path.exists(rule_file):
        try:
            rule_df = pq.read_table(rule_file).to_pandas()
            print(f"Found rule_scores.parquet with {len(rule_df)} frames.")
            
            if 'episode_index' in rule_df.columns and 'rule_score' in rule_df.columns:
                # Calculate mean rule score per episode
                ep_stats = rule_df.groupby('episode_index')[['rule_score', 'progress_sparse']].mean()
                print("Average per-episode metrics:")
                for ep_idx, row in ep_stats.iterrows():
                    rs = row.get('rule_score', float('nan'))
                    ps = row.get('progress_sparse', float('nan'))
                    print(f"  Episode {int(ep_idx)}: Rule Score = {rs:.3f} | Progress Sparse = {ps:.3f}")
            else:
                print("Missing standard columns 'episode_index' or 'rule_score' in rule_scores.parquet.")
        except Exception as e:
            print(f"Could not read rule_scores: {e}")
    else:
        print("No rule_scores.parquet found (this is normal for non-eval datasets).")

    # 3. Stats JSON
    print("\n--- 3. Data Statistics (stats.json) ---")
    stats_path = os.path.join(dataset_path, 'meta', 'stats.json')
    if os.path.exists(stats_path):
        try:
            with open(stats_path, 'r') as f:
                stats = json.load(f)
                
            print(f"Features recorded in stats: {list(stats.keys())}")
            # If action stats are present, print the rough bounds
            if 'action' in stats:
                print(f"  Action bounds (min): {np.array(stats['action'].get('min', [])).round(2).tolist()}")
                print(f"  Action bounds (max): {np.array(stats['action'].get('max', [])).round(2).tolist()}")
        except Exception as e:
            print(f"Error parsing stats.json: {e}")
    else:
        print("No stats.json found.")
        
    # 4. Movement Analysis
    print("\n--- 4. Episode lengths and movement analysis ---")
    try:
        dataset_arrow = ds.dataset(data_files, format="parquet")
        table = dataset_arrow.to_table(columns=['observation.state', 'episode_index'])
        df = table.to_pandas()
        
        total_moving = 0
        total_frames = 0
        
        unique_eps = sorted(df['episode_index'].unique())
        
        for ep in unique_eps:
            ep_data = df[df['episode_index'] == ep]
            states = np.stack(ep_data['observation.state'].values)
            num_frames = len(states)
            
            diffs = np.linalg.norm(np.diff(states, axis=0), axis=1)
            moving_frames = int(np.sum(diffs > 0.015)) # Threshold for movement
            stationary_frames = num_frames - 1 - moving_frames
            
            duration = num_frames / fps
            moving_duration = moving_frames / fps
            stationary_duration = stationary_frames / fps
            
            print(f'Episode {ep}: {duration:.2f}s (Moving: {moving_duration:.2f}s, Stationary: {stationary_duration:.2f}s)')
            
            total_moving += moving_frames
            total_frames += num_frames

        total_stationary = total_frames - len(unique_eps) - total_moving
        total_duration = total_frames / fps
        total_moving_duration = total_moving / fps
        total_stationary_duration = total_stationary / fps

        print("-" * 40)
        print(f'Total Time: {total_duration:.2f}s')
        print(f'Moving Time: {total_moving_duration:.2f}s ({total_moving_duration/total_duration*100:.1f}%)')
        print(f'Stationary Time: {total_stationary_duration:.2f}s ({total_stationary_duration/total_duration*100:.1f}%)')
    except Exception as e:
        print(f"Error during movement analysis: {e}")

    # 5. SARM Progress Data
    print("\n--- 5. SARM Progress Data (sarm_progress.parquet) ---")
    sarm_path = os.path.join(dataset_path, 'sarm_progress.parquet')
    if os.path.exists(sarm_path):
        try:
            sarm_df = pq.read_table(sarm_path).to_pandas()
            print(f"Found sarm_progress.parquet with {len(sarm_df)} frames.")
            if 'progress_dense' in sarm_df.columns:
                print("Dense Progress Stats:")
                print(f"  Min:  {sarm_df['progress_dense'].min():.6f}")
                print(f"  Max:  {sarm_df['progress_dense'].max():.6f}")
                print(f"  Mean: {sarm_df['progress_dense'].mean():.6f}")
                
                if 'episode_index' in sarm_df.columns and 'frame_index' in sarm_df.columns:
                    first_ep = sarm_df['episode_index'].min()
                    ep_data = sarm_df[sarm_df['episode_index'] == first_ep].sort_values('frame_index')
                    if len(ep_data) > 0:
                        print(f"\nEpisode {first_ep} Progress Sampling:")
                        sample = ep_data.iloc[::max(1, len(ep_data)//5)][['frame_index', 'progress_dense']]
                        for _, row in sample.iterrows():
                            print(f"  Frame {int(row['frame_index'])}: {row['progress_dense']:.4f}")
                        if len(ep_data) > 1 and int(sample.iloc[-1]['frame_index']) != int(ep_data.iloc[-1]['frame_index']):
                            last_row = ep_data.iloc[-1]
                            print(f"  Frame {int(last_row['frame_index'])}: {last_row['progress_dense']:.4f}")

            else:
                print("Missing 'progress_dense' column.")
        except Exception as e:
            print(f"Could not read sarm_progress.parquet: {e}")
    else:
        print("No sarm_progress.parquet found.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze.py <DATASET_DIRECTORY_PATH>")
        sys.exit(1)
        
    analyze_dataset(sys.argv[1])
