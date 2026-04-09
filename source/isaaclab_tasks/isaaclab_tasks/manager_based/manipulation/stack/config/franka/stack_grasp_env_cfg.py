from isaaclab.utils import configclass
from .stack_rl_env_cfg import FrankaStackRLEnvCfg
from ...grasp_reward import GraspRewardsCfg, GraspCurriculumCfg, GraspEventCfg

@configclass
class FrankaStackGraspCfg(FrankaStackRLEnvCfg):
    """Stack with grasping-specific reward for Franka parallel jaw gripper."""
    def __post_init__(self):
        super().__post_init__()
        self.rewards = GraspRewardsCfg()
        self.curriculum = GraspCurriculumCfg()
        self.events = GraspEventCfg()
