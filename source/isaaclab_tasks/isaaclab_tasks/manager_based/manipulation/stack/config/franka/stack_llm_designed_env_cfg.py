from isaaclab.utils import configclass
from .stack_rl_env_cfg import FrankaStackRLEnvCfg
from ...llm_designed_rewards import (
    LLMDesignedRewardsCfg, LLMDesignedCurriculumCfg, LLMDesignedEventCfg,
)

@configclass
class FrankaStackLLMDesignedCfg(FrankaStackRLEnvCfg):
    """Stack with LLM-designed-from-scratch rewards (no imported reward functions)."""
    def __post_init__(self):
        super().__post_init__()
        self.rewards = LLMDesignedRewardsCfg()
        self.curriculum = LLMDesignedCurriculumCfg()
        self.events = LLMDesignedEventCfg()
