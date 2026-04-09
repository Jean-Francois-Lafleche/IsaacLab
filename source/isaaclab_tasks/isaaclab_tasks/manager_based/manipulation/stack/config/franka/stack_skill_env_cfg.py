# Stack RL env with skill-decomposed curriculum

from isaaclab.utils import configclass
from .stack_rl_env_cfg import FrankaStackRLEnvCfg
from ...skill_curriculum import SkillRewardsCfg, SkillCurriculumCfg, SkillEventCfg


@configclass
class FrankaStackSkillCfg(FrankaStackRLEnvCfg):
    """Stack with skill-decomposed LLM curriculum."""
    def __post_init__(self):
        super().__post_init__()
        self.rewards = SkillRewardsCfg()
        self.curriculum = SkillCurriculumCfg()
        self.events = SkillEventCfg()
