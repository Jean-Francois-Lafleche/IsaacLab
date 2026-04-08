from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import FrankaCubeLiftEnvCfg
from isaaclab_tasks.manager_based.manipulation.lift.lift_adaptive_curriculum import (
    AdaptiveCurriculumCfg, AdaptiveEventCfg,
)

@configclass
class FrankaLiftAdaptiveCfg(FrankaCubeLiftEnvCfg):
    """LLM-in-the-loop adaptive curriculum."""
    def __post_init__(self):
        super().__post_init__()
        self.events = AdaptiveEventCfg()
        self.curriculum = AdaptiveCurriculumCfg()
