lerobot-teleoperate --robot.type=so101_follower --robot.port=COM5 --robot.id=so101_follower_left --teleop.type=so101_leader --teleop.port=COM4 --teleop.id=so101_leader_left

lerobot-teleoperate --robot.type=so101_follower --robot.port=COM6 --robot.id=so101_follower_right --teleop.type=so101_leader --teleop.port=COM7 --teleop.id=so101_leader_right

lerobot-teleoperate `
   --robot.type=bi_so100_follower `
   --robot.left_arm_port=COM5 `
   --robot.right_arm_port=COM6 `
   --robot.id=bimanual_follower `
   --teleop.type=bi_so100_leader `
   --teleop.left_arm_port=COM4 `
   --teleop.right_arm_port=COM7 `
   --teleop.id=bimanual_leader





lerobot-record --robot.type=so101_follower --robot.port=COM6 --robot.id=so101_follower_arm --robot.cameras="{ camera1: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, camera3: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" --display_data=true --dataset.repo_id="local/eval_Fold_the_rag0209TEST1" --dataset.single_task="Fold the rag" --dataset.num_episodes=50 --dataset.episode_time_s=100 --dataset.reset_time_s=5 --dataset.push_to_hub=false --policy.path="C:\Users\ccu\mujoco_ur5_graph\outputs\model\Fold-the-rag-transforms-only-expert=true\checkpoints\050000\pretrained_model" --policy.device=cuda --teleop.type=so101_leader --teleop.port=COM7 --teleop.id=so101_leader_arm


lerobot-record `
     --robot.type=bi_so100_follower `
     --robot.left_arm_port=COM5 `
     --robot.right_arm_port=COM6 `
     --robot.id=bimanual_follower `
     --robot.cameras='{"camera1": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}, "camera3": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}}' `
     --teleop.type=bi_so100_leader `
     --teleop.left_arm_port=COM4 `
     --teleop.right_arm_port=COM7 `
     --teleop.id=bimanual_leader `
     --display_data=true `
     --dataset.repo_id=local/full-fold-the-rag-parquet `
     --dataset.root=C:\Users\ccu\mujoco_ur5_graph\outputs\dataset\full-fold-the-rag-jpeg-parquet-0222 `
     --dataset.num_episodes=50 `
     --dataset.single_task="full fold the rag" `
     --dataset.video=false `
     --dataset.push_to_hub=false `
     --dataset.episode_time_s=180 `
     --dataset.reset_time_s=5


lerobot-record `
     --robot.type=bi_so100_follower `
     --robot.left_arm_port=COM5 `
     --robot.right_arm_port=COM6 `
     --robot.id=bimanual_follower `
     --robot.cameras='{"camera1": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}, "camera3": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}}' `
     --display_data=true `
     --dataset.repo_id=local/eval_Fold_the_rag0223TEST1 `
     --dataset.single_task="full Fold the rag" `
     --dataset.num_episodes=50 `
     --dataset.episode_time_s=300 `
     --dataset.reset_time_s=5 `
     --dataset.push_to_hub=false `
     --policy.path=C:\Users\ccu\mujoco_ur5_graph\outputs\model\040000\pretrained_model `
     --policy.device=cuda `
     --teleop.type=bi_so100_leader `
     --teleop.left_arm_port=COM4 `
     --teleop.right_arm_port=COM7 `
     --teleop.id=bimanual_leader


  $env:HF_LEROBOT_HOME='C:\Users\ccu\mujoco_ur5_graph\outputs'
  python -m lerobot.scripts.lerobot_edit_dataset `
    --repo_id "full-fold-the-rag-parquet-merged" `
    --operation.type merge `
    --operation.repo_ids "['full-fold-the-rag-parquet','full-fold-the-rag-parquet-c']"

lerobot-train   --dataset.root="/home/wy/outputs/full-fold-the-rag-jpeg-parquet"   --dataset.repo_id="local/full-Fold-the-rag-jpeg-parquet"   --policy.path="lerobot/smolvla_base"   --policy.device=cuda  --steps=50000 --num_workers=4   --output_dir="/home/wy/outputs/train/full-Fold-the-rag-jpeg-pa
rauet-vlm2.2b"   --job_name="full-Fold-the-rag"   --policy.repo_id="local/full-fold-the-rag-jpeg-parquet"   --policy.push_to_hub=false   --wandb.enable=false --dataset.imag
e_transforms.enable=true 

lerobot-train   --dataset.root="/home/wy/outputs/full-fold-the-rag-jpeg-parquet"   --dataset.repo_id="local/full-Fold-the-rag-jpeg-parquet"   --policy.path="lerobot/smolvla_base"   --policy.device=cuda  --steps=50000 --num_workers=4   --output_dir="/home/wy/outputs/train/full-Fold-the-rag-jpeg-pa
rauet-vlm2.2b"   --job_name="full-Fold-the-rag"   --policy.repo_id="local/full-fold-the-rag-jpeg-parquet"   --policy.push_to_hub=false   --wandb.enable=false --dataset.imag
e_transforms.enable=true --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-2.2B

$env:HF_LEROBOT_HOME='C:\Users\ccu\mujoco_ur5_graph\outputs'
python -m lerobot.scripts.lerobot_edit_dataset `
    --repo_id "full-fold-the-rag-parquet-merged0222" `
    --operation.type merge `
    --operation.repo_ids "['dataset/full-fold-the-rag-jpeg-parquet-0222','full-fold-the-rag-jpeg-parquet']"