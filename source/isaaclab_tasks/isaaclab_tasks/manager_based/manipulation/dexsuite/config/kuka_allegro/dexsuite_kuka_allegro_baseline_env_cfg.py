# Dexsuite Kuka Allegro - Baseline (NO curriculum)
# Disables the ADR curriculum to serve as baseline comparison.

from isaaclab.utils import configclass

from .dexsuite_kuka_allegro_env_cfg import (
    DexsuiteKukaAllegroLiftEnvCfg,
)


@configclass
class DexsuiteKukaAllegroLiftBaselineEnvCfg(DexsuiteKukaAllegroLiftEnvCfg):
    """Kuka Allegro Lift with NO curriculum — baseline for comparison."""

    def __post_init__(self):
        super().__post_init__()
        # Disable curriculum entirely
        self.curriculum = None
        # Set gravity to full immediately (the default ADR starts with zero gravity)
        self.events.variable_gravity.params["gravity_distribution_params"] = (
            (0.0, 0.0, -9.81),
            (0.0, 0.0, -9.81),
        )
