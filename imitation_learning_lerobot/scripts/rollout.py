import logging
import time
from contextlib import nullcontext
from dataclasses import asdict
from pprint import pformat

import re
import torch

from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.utils.utils import get_safe_torch_device
from lerobot.envs.utils import preprocess_observation

from imitation_learning_lerobot import configs
from imitation_learning_lerobot.envs import Env, EnvFactory

import numpy as np


@parser.wrap()
def main(cfg: EvalPipelineConfig):
    logging.info(pformat(asdict(cfg)))

    device = get_safe_torch_device(cfg.policy.device, log=True)

    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)
    policy.eval()

    env_cls = EnvFactory.get_strategies(cfg.env.type)
    env = env_cls(render_mode="human")

    max_steps = 500
    num_episodes = 100   # 跑 10 次
    success_count = 0

    for ep in range(num_episodes):
        observation, info = env.reset()
        step_count = 0

        # 清掉 policy 狀態（如果有）
        if hasattr(policy, "reset"):
            policy.reset()

        while True:
            obs = preprocess_observation(observation)
            obs = {k: obs[k].to(device, non_blocking=device.type == "cuda") for k in obs}
            obs["task"] = cfg.env.type

            with torch.inference_mode():
                action = policy.select_action(obs)

            action = action.to("cpu").numpy().flatten()
            print("action:", action)

            observation, _, terminated, truncated, info = env.step(action)
            env.render()
            time.sleep(1.0 / env.control_hz)

            step_count += 1

            if info.get("is_success", False) or terminated or truncated or step_count >= max_steps:
                print(f"[EP {ep+1}/{num_episodes} DONE] success={info.get('is_success', False)} "
                      f"terminated={terminated} truncated={truncated} steps={step_count}")
                if info.get("is_success", False):
                    success_count += 1
                break  # 結束當前 episode，進入下一次

    # 統計成功率
    success_rate = success_count / num_episodes
    print(f"總共 {num_episodes} 次，成功 {success_count} 次，成功率 = {success_rate:.2%}")


if __name__ == '__main__':
    main()
