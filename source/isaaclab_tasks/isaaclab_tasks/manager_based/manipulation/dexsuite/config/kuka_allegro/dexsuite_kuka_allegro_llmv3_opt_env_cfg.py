# Dexsuite Kuka Allegro - LLM Curriculum v3 (optimized for L40)
# 8192 envs, larger network, faster difficulty ramp, softer gravity curve

from isaaclab.utils import configclass

from .dexsuite_kuka_allegro_env_cfg import (
    DexsuiteKukaAllegroLiftEnvCfg,
)
from ...llm_curriculum_v3 import (
    LLMv3CurriculumCfg,
    LLMv3EventCfg,
)


@configclass
class DexsuiteKukaAllegroLiftLLMv3OptEnvCfg(DexsuiteKukaAllegroLiftEnvCfg):
    """Kuka Allegro Lift with LLM v3 — optimized for L40 GPU convergence."""

    def __post_init__(self):
        super().__post_init__()
        self.curriculum = LLMv3CurriculumCfg()
        self.events = LLMv3EventCfg()
        self.rewards.success.params["rot_std"] = None
        self.commands.object_pose.position_only = True
        self.commands.object_pose.ranges.pos_x = (-0.6, -0.5)
        self.commands.object_pose.ranges.pos_y = (-0.05, 0.05)
        self.commands.object_pose.ranges.pos_z = (0.35, 0.45)
        # 8192 envs for better GPU utilization
        self.scene.num_envs = 4096
