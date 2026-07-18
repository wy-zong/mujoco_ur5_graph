# 在推理主機 WSL 中執行。
# 先用 02_推理主機_wsl.sh attach，並確認四個 /dev/tty* 分別對應哪隻手臂。
# 以下 /dev/tty* 是範例，請改成實際結果。
6849
ROBOT_LEFT_PORT=/dev/ttyACM1
ROBOT_RIGHT_PORT=/dev/ttyACM0
TELEOP_LEFT_PORT=/dev/ttyACM3
TELEOP_RIGHT_PORT=/dev/ttyACM2
300 COM6
942 COM8
204 COM4
953 COM9

#找上面的方法類似
(new_mj) wy@DESKTOP-F1PB69R:~$ ls -l /dev/serial/by-id/ 
total 0
lrwxrwxrwx 1 root root 13 May  8 21:50 usb-1a86_USB_Single_Serial_5AAF219204-if00 -> ../../ttyACM4
lrwxrwxrwx 1 root root 13 May  8 21:17 usb-1a86_USB_Single_Serial_5AAF219300-if00 -> ../../ttyACM1
lrwxrwxrwx 1 root root 13 May  8 21:58 usb-1a86_USB_Single_Serial_5B42137942-if00 -> ../../ttyACM6
lrwxrwxrwx 1 root root 13 May  8 21:58 usb-1a86_USB_Single_Serial_5B42137953-if00 -> ../../ttyACM5

#執1行電腦要開才能開啟rerun
rerun --port 9876

lerobot-rollout \
       --strategy.type=dagger \
       --strategy.record_autonomous=true \
       --strategy.num_episodes=5 \
       --robot.type=bi_so_follower \
       --robot.left_arm_config.port="${ROBOT_LEFT_PORT}" \
       --robot.right_arm_config.port="${ROBOT_RIGHT_PORT}" \
       --robot.id=bimanual_follower \
       --teleop.type=bi_so_leader \
       --teleop.left_arm_config.port="${TELEOP_LEFT_PORT}" \
       --teleop.right_arm_config.port="${TELEOP_RIGHT_PORT}" \
       --teleop.id=bimanual_leader \
       --robot.left_arm_config.cameras='{"camera1": {"type": "opencv", "index_or_path": "/dev/video0", "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}, "camera3": {"type": "opencv", "index_or_path": "/dev/video4", "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}}' \
       --robot.right_arm_config.cameras='{"camera2": {"type": "opencv", "index_or_path": "/dev/video2", "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}}' \
       --display_data=false \
       --dataset.repo_id=local/rollout_bi_so101_flatten-and-fold-the-rag-then-place-pi05-folding-final-relative-all-linear-lora-inference.rtc.execution_horizon=1_0512EST1 \
       --dataset.single_task="flatten and fold the rag then place" \
       --dataset.video=true \
       --display_ip=192.168.50.161 \
       --display_port=9876 \
       --dataset.push_to_hub=false \
       --dataset.episode_time_s=300 \
       --dataset.reset_time_s=5 \
       --dataset.vcodec=h264 \
       --dataset.streaming_encoding=true \
       --dataset.encoder_threads=2 \
       --policy.path="wuc1/bi_so101_flatten-and-fold-the-rag-then-place-pi05-folding-final-relative-all-linear-lora" \
       --fps=15 \
       --dataset.fps=15 \
       --inference.type=rtc \
       --inference.rtc.execution_horizon=1 \
       --inference.rtc.max_guidance_weight=10.0 \
       --rename_map='{"observation.images.left_camera1": "observation.images.camera1", "observation.images.left_camera3": "observation.images.camera3", "observation.images.right_camera2": "observation.images.camera2"}'\
