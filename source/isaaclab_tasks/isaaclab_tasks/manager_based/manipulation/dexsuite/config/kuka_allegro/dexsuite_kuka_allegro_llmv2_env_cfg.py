# Dexsuite Kuka Allegro - LLM Curriculum v2
# Per-env smooth interpolation of all parameters. No ADR.

from isaaclab.utils import configclass

from .dexsuite_kuka_allegro_env_cfg import (
    DexsuiteKukaAllegroLiftEnvCfg,
)
from ...llm_curriculum_v2 import (
    LLMv2CurriculumCfg,
    LLMv2EventCfg,
)


@configclass
class DexsuiteKukaAllegroLiftLLMv2EnvCfg(DexsuiteKukaAllegroLiftEnvCfg):
    """Kuka Allegro Lift with LLM Curriculum v2 — smooth per-env difficulty."""

    def __post_init__(self):
        super().__post_init__()
        self.curriculum = LLMv2CurriculumCfg()
        self.events = LLMv2EventCfg()
        self.rewards.success.params["rot_std"] = None
        self.commands.object_pose.position_only = True
        # Start with easy goals
        self.commands.object_pose.ranges.pos_x = (-0.6, -0.5)
        self.commands.object_pose.ranges.pos_y = (-0.05, 0.05)
        self.commands.object_pose.ranges.pos_z = (0.35, 0.45)
