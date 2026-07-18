lerobot-record `
        --robot.type=bi_so_follower `
        --robot.left_arm_config.port=COM5 `
        --robot.right_arm_config.port=COM8 `
        --robot.id=bimanual_follower `
        --teleop.type=bi_so_leader `
        --teleop.left_arm_config.port=COM4 `
        --teleop.right_arm_config.port=COM9 `
        --teleop.id=bimanual_leader `
        --robot.left_arm_config.cameras='{"camera1": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}, "camera3":{"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}, "camera2": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}}' `
        --display_data=true `
        --dataset.repo_id=wuc1/bi_so101_flatten-the-rag-debug`
        --dataset.single_task="flatten the rag" `
        --dataset.push_to_hub=false

使用left_arm_config.cameras可以在
        --robot.id=bimanual_follower `
        --teleop.type=bi_so_leader 
的情況下正常蒐集資料
但影片名稱會被加上前綴變成observation.images.left_camera1

lerobot-record `
        --robot.type=bi_so_follower `
        --robot.left_arm_config.port=COM5 `
        --robot.right_arm_config.port=COM8 `
        --robot.id=bimanual_follower `
        --teleop.type=bi_so_leader `
        --teleop.left_arm_config.port=COM4 `
        --teleop.right_arm_config.port=COM9 `
        --teleop.id=bimanual_leader `
        --robot.cameras='{"camera1": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}, "camera3": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}, "camera2": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}}' `
        --dataset.repo_id=wuc1/bi_so101_flatten-the-rag-debug`
        --dataset.single_task="flatten the rag" `
        --dataset.push_to_hub=false 

使用robot.cameras則會遇到錯誤
log如下
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Users\ccu\anaconda3\envs\new_mj\Scripts\lerobot-record.exe\__main__.py", line 5, in <module>
  File "C:\Users\ccu\mujoco_ur5_graph\lerobot\src\lerobot\scripts\lerobot_record.py", line 605, in main
    record()
  File "C:\Users\ccu\mujoco_ur5_graph\lerobot\src\lerobot\configs\parser.py", line 232, in wrapper_inner
    cfg = draccus.parse(config_class=argtype, config_path=config_path, args=cli_args)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ccu\anaconda3\envs\new_mj\Lib\site-packages\draccus\argparsing.py", line 211, in parse
    return parser.parse_args(args)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ccu\anaconda3\envs\new_mj\Lib\site-packages\draccus\argparsing.py", line 102, in parse_args
    args, _ = self.parse_known_args(args, namespace, is_parse_args=True)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ccu\anaconda3\envs\new_mj\Lib\site-packages\draccus\argparsing.py", line 136, in parse_known_args
    parsed_t = self._postprocessing(parsed_args)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ccu\anaconda3\envs\new_mj\Lib\site-packages\draccus\argparsing.py", line 180, in _postprocessing
    cfg = decoding.decode(self.config_class, deflat_d)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ccu\anaconda3\envs\new_mj\Lib\site-packages\draccus\parsers\registry_utils.py", line 78, in wrapper
    return base_func(*args, **kw)
           ^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ccu\anaconda3\envs\new_mj\Lib\site-packages\draccus\parsers\decoding.py", line 48, in decode
    return get_decoding_fn(cls)(raw_value, ())  # type: ignore
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ccu\anaconda3\envs\new_mj\Lib\site-packages\draccus\parsers\decoding.py", line 135, in decode_dataclass
    raise e
  File "C:\Users\ccu\anaconda3\envs\new_mj\Lib\site-packages\draccus\parsers\decoding.py", line 133, in decode_dataclass
    field_value = get_decoding_fn(field_type)(raw_value, (*path, name))
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ccu\anaconda3\envs\new_mj\Lib\site-packages\draccus\parsers\decoding.py", line 201, in decode_choice_class
    return decode_dataclass(subcls, raw_value, path)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\ccu\anaconda3\envs\new_mj\Lib\site-packages\draccus\parsers\decoding.py", line 149, in decode_dataclass
    raise DecodingError(path, f"The fields {formatted_keys} are not valid for {stringify_type(cls)}")
draccus.utils.DecodingError: `robot`: The fields `cameras` are not valid for BiSOFollowerConfig


lerobot-record `
  --robot.type=bi_so_follower `
  --robot.left_arm_config.port=COM5 `
  --robot.right_arm_config.port=COM8 `
  --robot.id=bimanual_follower `
  --teleop.type=bi_so_leader `
  --teleop.left_arm_config.port=COM4 `
  --teleop.right_arm_config.port=COM9 `
  --teleop.id=bimanual_leader `
  --robot.cameras='{"camera1": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}' `
  --dataset.repo_id=local/test `
  --dataset.single_task="test task" `
  --dataset.push_to_hub=false


lerobot-record `
  --robot.type=bi_so_follower `
  --robot.left_arm_config.port=COM5 `
  --robot.right_arm_config.port=COM8 `
  --robot.id=bimanual_follower `
  --teleop.type=bi_so_leader `
  --teleop.left_arm_config.port=COM4 `
  --teleop.right_arm_config.port=COM9 `
  --teleop.id=bimanual_leader `
  --robot.left_arm_config.cameras='{"camera1": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}' `
  --dataset.repo_id=local/test `
  --dataset.single_task="test task" `
  --dataset.push_to_hub=false

  