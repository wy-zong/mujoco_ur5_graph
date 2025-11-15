export MUJOCO_GL=egl
python /home/wuc120/imitation_learning_lerobot/imitation_learning_lerobot/scripts/rollout.py \
  --policy.path=/home/wuc120/imitation_learning_lerobot/outputs/model/050000/pretrained_model \
  --env.type=grasp_cloth \
  --policy.device=cuda \
  --policy.use_amp=true



  /home/wuc120/imitation_learning_lerobot/imitation_learning_lerobot/scripts/1103rollout.py
export MUJOCO_GL=egl
python /home/wuc120/imitation_learning_lerobot/imitation_learning_lerobot/scripts/1103rollout.py \
  --env.discover_packages_path=imitation_learning_lerobot.configs \
  --policy.path=/home/wuc120/outputs/models/smolvla_grasp_cloth/checkpoints/000400/pretrained_model \
  --env.type=pick_an_place \
  --policy.device=cuda \
  --policy.use_amp=true



export MUJOCO_GL=egl
python /home/wuc120/imitation_learning_lerobot/imitation_learning_lerobot/scripts/rollout.py \
  --policy.path=/home/wuc120/imitation_learning_lerobot/outputs/model/050000/pretrained_model \
  --env.type=grasp_cloth \
  --policy.device=cuda \
  --policy.use_amp=true