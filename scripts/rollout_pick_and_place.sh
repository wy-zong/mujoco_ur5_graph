export MUJOCO_GL=egl
python /home/wuc120/imitation_learning_lerobot/imitation_learning_lerobot/scripts/rollout.py \
  --policy.path=/home/wuc120/imitation_learning_lerobot/outputs/model/pick_and_place_20250910_213352_batch64_10000_n_obs=5/checkpoints/010000/pretrained_model \
  --env.type=pick_and_place \
  --policy.device=cuda \
  --policy.use_amp=true