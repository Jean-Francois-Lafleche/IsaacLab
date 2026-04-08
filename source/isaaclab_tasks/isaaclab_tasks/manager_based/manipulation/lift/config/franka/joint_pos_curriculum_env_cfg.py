# Copyright (c) 2024, Curriculum Experiment
# Franka Lift Cube with curriculum — inherits from standard but overrides events and curriculum.

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import (
    FrankaCubeLiftEnvCfg,
)
from isaaclab_tasks.manager_based.manipulation.lift.lift_curriculum_cfg import (
    LiftCurriculumCfg,
    LiftCurriculumEventCfg,
)


@configclass
class FrankaCubeLiftCurriculumEnvCfg(FrankaCubeLiftEnvCfg):
    """Franka cube lift with progressive curriculum."""

    def __post_init__(self):
        super().__post_init__()

        # Override events to start with easy spawn range
        self.events = LiftCurriculumEventCfg()

        # Override curriculum with our progressive version
        self.curriculum = LiftCurriculumCfg()

        # Use same command ranges initially — curriculum will narrow them at start
        # The curriculum function will modify these dynamically
