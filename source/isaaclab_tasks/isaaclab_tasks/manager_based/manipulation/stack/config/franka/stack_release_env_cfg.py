from isaaclab.utils import configclass
from .stack_rl_env_cfg import FrankaStackRLEnvCfg
from ...release_reward import ReleaseRewardsCfg, ReleaseCurriculumCfg, ReleaseEventCfg

@configclass
class FrankaStackReleaseCfg(FrankaStackRLEnvCfg):
    """Release skill: open gripper, retract arm, rest. Cube pre-stacked."""
    def __post_init__(self):
        super().__post_init__()
        self.rewards = ReleaseRewardsCfg()
        self.curriculum = ReleaseCurriculumCfg()
        self.events = ReleaseEventCfg()

@configclass
class FrankaStackReleaseCfg_PLAY(FrankaStackReleaseCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
