# Dexsuite Kuka Allegro - LLM-Crafted Progressive Curriculum
# No ADR — pure task-specific stages designed by LLM analysis.

from isaaclab.utils import configclass

from .dexsuite_kuka_allegro_env_cfg import (
    DexsuiteKukaAllegroLiftEnvCfg,
)
from ...llm_curriculum import (
    LLMCurriculumCfg,
    LLMCurriculumEventCfg,
)


@configclass
class DexsuiteKukaAllegroLiftLLMCurriculumEnvCfg(DexsuiteKukaAllegroLiftEnvCfg):
    """Kuka Allegro Lift with LLM-crafted progressive curriculum (no ADR)."""

    def __post_init__(self):
        super().__post_init__()
        # Replace ADR curriculum with LLM-crafted stages
        self.curriculum = LLMCurriculumCfg()
        # Replace events with LLM curriculum events (easy start)
        self.events = LLMCurriculumEventCfg()
        # The LLM curriculum handles gravity scheduling itself
        # Make success reward not consider orientation (lift mode)
        self.rewards.success.params["rot_std"] = None
        self.commands.object_pose.position_only = True
        # Start with close goals for Stage 0
        self.commands.object_pose.ranges.pos_x = (-0.6, -0.5)
        self.commands.object_pose.ranges.pos_y = (-0.05, 0.05)
        self.commands.object_pose.ranges.pos_z = (0.35, 0.45)
