from isaaclab.utils import configclass
from .stack_rl_env_cfg import FrankaStackRLEnvCfg
from ...llm_adaptive_skill import (
    LLMAdaptiveRewardsCfg, LLMAdaptiveCurriculumCfg, LLMAdaptiveEventCfg,
)

@configclass
class FrankaStackLLMAdaptiveCfg(FrankaStackRLEnvCfg):
    """Stack with LLM-in-the-loop adaptive skill training."""
    def __post_init__(self):
        super().__post_init__()
        self.rewards = LLMAdaptiveRewardsCfg()
        self.curriculum = LLMAdaptiveCurriculumCfg()
        self.events = LLMAdaptiveEventCfg()
