# Dexsuite Kuka Allegro - LLM Curriculum v4 (sticky retry)
# Failed envs reset to same state; succeeded envs get fresh conditions.

from isaaclab.utils import configclass

from .dexsuite_kuka_allegro_env_cfg import (
    DexsuiteKukaAllegroLiftEnvCfg,
)
from ...llm_curriculum_v4 import (
    LLMv4CurriculumCfg,
    LLMv4EventCfg,
)


@configclass
class DexsuiteKukaAllegroLiftLLMv4EnvCfg(DexsuiteKukaAllegroLiftEnvCfg):
    """Kuka Allegro Lift with LLM v4 — sticky retry on failure."""

    def __post_init__(self):
        super().__post_init__()
        self.curriculum = LLMv4CurriculumCfg()
        self.events = LLMv4EventCfg()
        self.rewards.success.params["rot_std"] = None
        self.commands.object_pose.position_only = True
        self.commands.object_pose.ranges.pos_x = (-0.6, -0.5)
        self.commands.object_pose.ranges.pos_y = (-0.05, 0.05)
        self.commands.object_pose.ranges.pos_z = (0.35, 0.45)
        self.scene.num_envs = 4096
