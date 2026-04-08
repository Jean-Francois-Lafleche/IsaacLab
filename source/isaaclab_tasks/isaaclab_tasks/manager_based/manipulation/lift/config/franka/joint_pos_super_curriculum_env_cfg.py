# Copyright (c) 2024, Super Curriculum Experiment
# Franka Lift Cube with Super Curriculum — adaptive failure-aware sampling.

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import (
    FrankaCubeLiftEnvCfg,
)
from isaaclab_tasks.manager_based.manipulation.lift.lift_super_curriculum import (
    SuperCurriculumCfg,
    SuperCurriculumEventCfg,
)


@configclass
class FrankaCubeLiftSuperCurriculumEnvCfg(FrankaCubeLiftEnvCfg):
    """Franka cube lift with Super Curriculum — progressive + adaptive failure sampling."""

    def __post_init__(self):
        super().__post_init__()

        # Override events with adaptive reset
        self.events = SuperCurriculumEventCfg()

        # Override curriculum with super curriculum
        self.curriculum = SuperCurriculumCfg()
