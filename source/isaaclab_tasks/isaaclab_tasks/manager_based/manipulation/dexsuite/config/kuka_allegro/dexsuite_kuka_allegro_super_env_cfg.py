# Dexsuite Kuka Allegro - Super Curriculum env config
# ADR + adaptive failure-aware object reset sampling.

from isaaclab.utils import configclass

from .dexsuite_kuka_allegro_env_cfg import (
    DexsuiteKukaAllegroLiftEnvCfg,
)
from ...dexsuite_super_curriculum import (
    DexsuiteSuperCurriculumCfg,
    DexsuiteSuperEventCfg,
)


@configclass
class DexsuiteKukaAllegroLiftSuperCurriculumEnvCfg(DexsuiteKukaAllegroLiftEnvCfg):
    """Kuka Allegro Lift with ADR + adaptive failure-aware sampling."""

    def __post_init__(self):
        super().__post_init__()
        # Override curriculum with super curriculum (ADR + adaptive)
        self.curriculum = DexsuiteSuperCurriculumCfg()
        # Override events with adaptive reset
        self.events = DexsuiteSuperEventCfg()
        # Re-apply the ADR curriculum params that __post_init__ sets
        if self.curriculum is not None:
            self.curriculum.adr.params["pos_tol"] = self.rewards.success.params["pos_std"] / 2
            self.curriculum.adr.params["rot_tol"] = None  # lift mode
            self.rewards.success.params["rot_std"] = None
