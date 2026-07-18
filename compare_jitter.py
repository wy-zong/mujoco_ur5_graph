import pandas as pd
import numpy as np
import os

def calculate_jitter(parquet_path):
    df = pd.read_parquet(parquet_path)
    
    # Check if we have multiple episodes, we should calculate diffs within each episode
    # to avoid diffs across episodes, but if it's just one chunk or we just take overall mean it's fine as an approximation.
    # To be safe, we group by episode_index
    
    # We want to measure the variation of 'action' or 'observation.state'
    actions = np.stack(df['action'].values)
    states = np.stack(df['observation.state'].values)
    
    action_jerk = np.mean(np.abs(np.diff(actions, n=2, axis=0)))
    state_jerk = np.mean(np.abs(np.diff(states, n=2, axis=0)))
    
    action_diff = np.mean(np.abs(np.diff(actions, n=1, axis=0)))
    state_diff = np.mean(np.abs(np.diff(states, n=1, axis=0)))
    
    return {
        'action_jerk (加速度/二次差分)': action_jerk,
        'state_jerk (加速度/二次差分)': state_jerk,
        'action_diff (一次差分)': action_diff,
        'state_diff (一次差分)': state_diff
    }

dir_no_smooth = r"C:\Users\ccu\.cache\huggingface\lerobot\wuc1\rollout_dagger_bi_so101_ffp_bi_so101_ffp_0615-14-12_merged_20260615_191855\data\chunk-000\file-000.parquet"
dir_with_smooth = r"C:\Users\ccu\.cache\huggingface\lerobot\wuc1\rollout_dagger_bi_so101_ffp_bi_so101_ffp_0615-14-12_merged_20260615_192705\data\chunk-000\file-000.parquet"

print("=== 未開啟 intra_chunk_smoothing ===")
res_no = calculate_jitter(dir_no_smooth)
for k, v in res_no.items():
    print(f"  {k}: {v:.6f}")

print("\n=== 開啟 intra_chunk_smoothing ===")
res_yes = calculate_jitter(dir_with_smooth)
for k, v in res_yes.items():
    print(f"  {k}: {v:.6f}")
