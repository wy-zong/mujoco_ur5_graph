from dataclasses import dataclass, field

from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE
from lerobot.envs import EnvConfig
from lerobot.configs.types import FeatureType, PolicyFeature


@EnvConfig.register_subclass("so101_joint_control")
@dataclass
class SO101JointControlEnvConfig(EnvConfig):
    """
    SO101 Joint Control Environment Config
    專為測試訓練好的模型設計（smolvla_so101_e20 等）
    """
    task: str = "so101_joint_control"
    fps: int = 30  # 匹配真機控制頻率
    episode_length: int = 500
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "observation.state": OBS_STATE,
            "observation.images.camera1": f"{OBS_IMAGES}.camera1",
            "observation.images.camera3": f"{OBS_IMAGES}.camera3",
        }
    )

    def __post_init__(self):
        # 設定完整的特徵定義
        self.features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(6,))
        self.features["observation.images.camera1"] = PolicyFeature(type=FeatureType.VISUAL, shape=(480, 640, 3))
        self.features["observation.images.camera3"] = PolicyFeature(type=FeatureType.VISUAL, shape=(480, 640, 3))

    @property
    def gym_kwargs(self) -> dict:
        return {
            "render_mode": self.render_mode,
            "max_episode_steps": self.episode_length,
        }
