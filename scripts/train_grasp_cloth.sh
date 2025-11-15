python /home/wuc120/imitation_learning_lerobot/lerobot/src/lerobot/scripts/train.py \
  --output_dir=outputs/models/smolvla_grasp_cloth \
  --policy.type=smolvla \
  --dataset.repo_id=grasp_cloth \
  --dataset.root=/home/wuc120/imitation_learning_lerobot/outputs/datasets/grasp_cloth \
  --wandb.enable=false \
  --steps=2000 \
  --log_freq=200 \
  --save_freq=200 \
  --batch_size=8

python /home/wuc120/imitation_learning_lerobot/lerobot/src/lerobot/scripts/train.py \
  --output_dir=outputs/models/smolvla_grasp_cloth \
  --policy.type=smolvla \
  --dataset.repo_id=grasp_cloth \
  --dataset.root=/home/wy/imitation_learning_lerobot/outputs/datasets/grasp_cloth \
  --wandb.enable=false \
  --steps=50000 \
  --resume=True \
  --log_freq=200 \
  --save_freq=2000 \
  --batch_size=8 \
  --policy.push_to_hub=false \
  --policy.repo_id=local_only \
  --config_path=/home/wuc120/outputs/models/smolvla_grasp_cloth/checkpoints/002000/pretrained_model/train_config.json \
  --resume=true 
