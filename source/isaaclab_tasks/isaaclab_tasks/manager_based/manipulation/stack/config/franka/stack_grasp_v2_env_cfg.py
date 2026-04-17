from isaaclab.utils import configclass
from .stack_rl_env_cfg import FrankaStackRLEnvCfg
from ...grasp_reward_v2 import GraspV2RewardsCfg, GraspV2CurriculumCfg, GraspV2EventCfg

@configclass
class FrankaStackGraspV2Cfg(FrankaStackRLEnvCfg):
    """Stack with distribution-based curriculum and robot-rest goal state."""
    def __post_init__(self):
        super().__post_init__()
        self.rewards = GraspV2RewardsCfg()
        self.curriculum = GraspV2CurriculumCfg()
        self.events = GraspV2EventCfg()

@configclass
class FrankaStackGraspV2Cfg_PLAY(FrankaStackGraspV2Cfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
