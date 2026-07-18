from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 將本地資料集加載並直接上傳到 Hugging Face，
# 這會一併上傳影片、Parquet 資料，並自動產生包含 LeRobot 標籤的 README.md
dataset = LeRobotDataset(
    repo_id="wuc1/rollout_bi_so101_ffp_0615-14-12-dagger_merged3cam_discrete_state_20260621_192016",
    root=r"C:\Users\ccu\.cache\huggingface\lerobot\wuc1\rollout_bi_so101_ffp_0615-14-12-dagger_merged3cam_discrete_state_20260621_192016"
)
dataset.push_to_hub()

dataset2 = LeRobotDataset(
    repo_id="wuc1/rollout_dagger_bi_so101_ffp_0615-14-12-dagger_merged3cam_20260619_014303",
    root=r"C:\Users\ccu\.cache\huggingface\lerobot\wuc1\rollout_dagger_bi_so101_ffp_0615-14-12-dagger_merged3cam_20260619_014303"
)
dataset2.push_to_hub()

dataset3= LeRobotDataset(
    repo_id="wuc1/rollout_bi_so101_ffp_0615-14-12-dagger_merged3cam_nopolicyaction_no_use_smoothing",
    root=r"C:\Users\ccu\.cache\huggingface\lerobot\wuc1\rollout_bi_so101_ffp_0615-14-12-dagger_merged3cam_nopolicyaction_no_use_smoothing"
)
dataset3.push_to_hub()