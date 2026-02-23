import pandas as pd
from pathlib import Path

# Check source dataset
src_path = Path(r"C:\Users\ccu\mujoco_ur5_graph\outputs\full-fold-the-rag\data\chunk-000\file-000.parquet")
print(f"Checking source dataset: {src_path}")
src_df = pd.read_parquet(src_path)
print(f"Columns: {src_df.columns.tolist()}")
print(f"Shape: {src_df.shape}")

# Check if camera columns exist and have data
if 'observation.images.camera1' in src_df.columns:
    print(f"\ncamera1 column exists")
    print(f"First value type: {type(src_df['observation.images.camera1'].iloc[0])}")
    print(f"First value: {src_df['observation.images.camera1'].iloc[0]}")
else:
    print("\nWARNING: camera1 column NOT found in source!")

# Check merged dataset
merged_path = Path(r"C:\Users\ccu\mujoco_ur5_graph\outputs\full-fold-the-rag-merged\data\chunk-000\file-000.parquet")
print(f"\n{'='*60}")
print(f"Checking merged dataset: {merged_path}")
merged_df = pd.read_parquet(merged_path)
print(f"Columns: {merged_df.columns.tolist()}")
print(f"Shape: {merged_df.shape}")

# Check if camera columns exist and have data
if 'observation.images.camera1' in merged_df.columns:
    print(f"\ncamera1 column exists")
    print(f"First value type: {type(merged_df['observation.images.camera1'].iloc[0])}")
    print(f"First value: {merged_df['observation.images.camera1'].iloc[0]}")
else:
    print("\nWARNING: camera1 column NOT found in merged dataset!")
