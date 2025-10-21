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
    observation, info = env.reset()

    observation, info = env.reset()

    step_count = 0
    max_steps = 200  # 你原本的上限

    while True:
        obs = preprocess_observation(observation)
        obs = {k: obs[k].to(device, non_blocking=device.type == "cuda") for k in obs}
        obs["task"] = cfg.env.type
        # obs["task"] = [ (cfg.env.type if isinstance(cfg.env.type, str) else str(cfg.env.type)) + ("\n" if not str(cfg.env.type).endswith("\n") else "") ] * next((v.shape[0] for v in obs.values() if isinstance(v, torch.Tensor) and v.ndim>0), 1)


        with torch.inference_mode():
            action = policy.select_action(obs)

        action = action.to("cpu").numpy().flatten()
        print("action:", action)

        # 這裡把 terminated / truncated 接回來
        observation, _, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(1.0 / env.control_hz)

        step_count += 1

        # === 重置條件：成功 或 逾時(terminated/truncated) 或 自訂步數上限 ===
        if info.get("is_success", False) or terminated or truncated or step_count >= max_steps:
            print(f"[EP DONE] success={info.get('is_success', False)} "
                f"terminated={terminated} truncated={truncated} steps={step_count}")
            observation, info = env.reset()
            # 2) ★ 連做 K 次零動作 step，讓 frame stack 完全由「新 episode」填滿
            K = 20  # 典型堆疊長度；如果你知道實際值就用實際值
            zero = np.zeros(4, dtype=np.float32)
            for _ in range(K):
                observation, _, _, _, info = env.step(zero)

            # 3) （可選）把 policy/RNN 狀態也清掉，避免舊上下文殘留
            if hasattr(policy, "reset"):
                policy.reset()

            step_count = 0


if __name__ == '__main__':
    main()
