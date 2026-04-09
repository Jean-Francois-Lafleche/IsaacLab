# Stack RL env configs with reverse curriculum

from isaaclab.utils import configclass
from .stack_rl_env_cfg import FrankaStackRLEnvCfg
from ...reverse_curriculum import (
    ReverseStaticCurriculumCfg,
    ReverseAdaptiveCurriculumCfg,
    ReverseEventCfg,
)


@configclass
class FrankaStackReverseDetailedCfg(FrankaStackRLEnvCfg):
    """Stack with reverse curriculum — start near success, work backwards."""
    def __post_init__(self):
        super().__post_init__()
        self.curriculum = ReverseStaticCurriculumCfg()
        self.events = ReverseEventCfg()


@configclass
class FrankaStackReverseAdaptiveCfg(FrankaStackRLEnvCfg):
    """Stack with LLM-in-the-loop reverse curriculum."""
    def __post_init__(self):
        super().__post_init__()
        self.curriculum = ReverseAdaptiveCurriculumCfg()
        self.events = ReverseEventCfg()
