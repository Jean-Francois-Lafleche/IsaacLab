from isaaclab.utils import configclass
from .stack_rl_env_cfg import FrankaStackRLEnvCfg
from ...sequential_skills import SequentialRewardsCfg, SequentialCurriculumCfg, SequentialEventCfg


@configclass
class FrankaStackSequentialCfg(FrankaStackRLEnvCfg):
    """Sequential skill training: one reward at a time, per-skill curriculum."""
    def __post_init__(self):
        super().__post_init__()
        self.rewards = SequentialRewardsCfg()
        self.curriculum = SequentialCurriculumCfg()
        self.events = SequentialEventCfg()


@configclass
class FrankaStackSequentialCfg_PLAY(FrankaStackSequentialCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
