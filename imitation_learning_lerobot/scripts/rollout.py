import logging
import time
from contextlib import nullcontext
from dataclasses import asdict
from pprint import pformat

import re
import torch
import einops

from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.utils.utils import get_safe_torch_device
from lerobot.envs.utils import preprocess_observation

from imitation_learning_lerobot import configs
from imitation_learning_lerobot.envs import Env, EnvFactory

import numpy as np

# 放在最頂端、任何 @parser.wrap() 與 main() 呼叫之前
import imitation_learning_lerobot.configs.grasp_cloth_env_config
import imitation_learning_lerobot.configs.so101_joint_control_env_config


def preprocess_so101_observation(observations: dict) -> dict:
    """
    SO101 專用觀測預處理函數
    將環境輸出的 pixels/agent_pos 格式轉換為模型預期的 observation.images.*/observation.state 格式
    """
    return_observations = {}
    
    # # 處理 agent_pos 或 observation.state（關節角度）
    # if 'agent_pos' in observations:
    #     state = torch.from_numpy(observations['agent_pos']).float()
    #     if state.dim() == 1:
    #         state = state.unsqueeze(0)
    #     return_observations['observation.state'] = state
    # elif 'observation.state' in observations:
    #     state = torch.from_numpy(observations['observation.state']).float()
    #     if state.dim() == 1:
    #         state = state.unsqueeze(0)
    #     return_observations['observation.state'] = state


    state = torch.from_numpy(observations['observation.state']).float()
    if state.dim() == 1:
        state = state.unsqueeze(0)
    return_observations['observation.state'] = state
    
    # 處理 pixels 格式（環境輸出）-> 轉換為 observation.images.*
    if 'pixels' in observations and isinstance(observations['pixels'], dict):
        for cam_name, img in observations['pixels'].items():
            img_tensor = torch.from_numpy(img)
            if img_tensor.ndim == 3:
                img_tensor = img_tensor.unsqueeze(0)
            img_tensor = einops.rearrange(img_tensor, "b h w c -> b c h w").contiguous()
            img_tensor = img_tensor.type(torch.float32)
            img_tensor /= 255.0
            return_observations[f'observation.images.{cam_name}'] = img_tensor
    
    # 處理已經是 observation.images.* 格式的情況（備用）
    for key in observations:
        if key.startswith('observation.images.'):
            img = observations[key]
            img_tensor = torch.from_numpy(img)
            if img_tensor.ndim == 3:
                img_tensor = img_tensor.unsqueeze(0)
            img_tensor = einops.rearrange(img_tensor, "b h w c -> b c h w").contiguous()
            img_tensor = img_tensor.type(torch.float32)
            img_tensor /= 255.0
            return_observations[key] = img_tensor
    
    return return_observations



@parser.wrap()
def main(cfg: EvalPipelineConfig):
    logging.info(pformat(asdict(cfg)))

    device = get_safe_torch_device(cfg.policy.device, log=True)

    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)
    policy.eval()
    
    # 初始化 preprocessor 和 postprocessor (關鍵：處理觀測正規化和 action unnormalization)
    # 覆蓋設備設定以匹配當前可用設備
    preprocessor_overrides = {
        "device_processor": {"device": str(device)},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    print(f"[INFO] Initialized preprocessor and postprocessor for action unnormalization")
    
    # 初始化 tokenizer 用於 VLA 模型
    tokenizer = None
    # task_prompt = "Pick up the red cube and place it in the blue zone\n"
    task_prompt = "so101_joint_control\n"
    lang_tokens = None
    lang_attention_mask = None
    
    if cfg.env.type == "so101_joint_control":
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
            # 預先 tokenize 任務描述
            tokenized = tokenizer(
                task_prompt, 
                return_tensors="pt", 
                padding="max_length", 
                max_length=48,
                truncation=True
            )
            lang_tokens = tokenized["input_ids"].to(device)
            lang_attention_mask = tokenized["attention_mask"].bool().to(device)
            print(f"[INFO] Tokenized task: {task_prompt.strip()}")
            print(f"[INFO] Token IDs shape: {lang_tokens.shape}")
        except Exception as e:
            print(f"[WARN] Failed to initialize tokenizer: {e}")

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
            # 根據環境類型選擇預處理函數
            if cfg.env.type == "so101_joint_control":
                obs = preprocess_so101_observation(observation)
                # 添加任務描述字串（給 TokenizerProcessorStep 使用，如 SmolVLA）
                obs["task"] = task_prompt
                # 同時添加語言 tokens（給沒有 TokenizerProcessorStep 的模型使用，如 ACT）
                if lang_tokens is not None:
                    obs["observation.language.tokens"] = lang_tokens
                    obs["observation.language.attention_mask"] = lang_attention_mask
            else:
                obs = preprocess_observation(observation)
            
            # 使用 preprocessor 正規化觀測（包含移動到設備）
            try:
                obs = preprocessor(obs)

                with torch.inference_mode():
                    action = policy.select_action(obs)

                # 使用 postprocessor unnormalize action（關鍵！）
                action = postprocessor(action)
                action = action.to("cpu").numpy().flatten()
                print(f"[Step {step_count}] action:", action)
            except Exception as e:
                print(f"[ERROR at step {step_count}] {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                break

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
