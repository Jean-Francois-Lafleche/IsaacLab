# Three LLM curriculum variations for Franka Lift Cube

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import FrankaCubeLiftEnvCfg
from isaaclab_tasks.manager_based.manipulation.lift.lift_llm_curricula import (
    DetailedCurriculumCfg, HighlyDetailedCurriculumCfg, SuperDetailedCurriculumCfg,
    DetailedEventCfg, SuperDetailedEventCfg,
)


@configclass
class FrankaLiftDetailedCfg(FrankaCubeLiftEnvCfg):
    """Detailed: 4 scheduled params (spawn, goal range, goal height, penalties)"""
    def __post_init__(self):
        super().__post_init__()
        self.events = DetailedEventCfg()
        self.curriculum = DetailedCurriculumCfg()


@configclass
class FrankaLiftHighlyDetailedCfg(FrankaCubeLiftEnvCfg):
    """Highly Detailed: 7 params (+ reach std, lift threshold, action penalty shaping)"""
    def __post_init__(self):
        super().__post_init__()
        self.events = DetailedEventCfg()
        self.curriculum = HighlyDetailedCurriculumCfg()


@configclass
class FrankaLiftSuperDetailedCfg(FrankaCubeLiftEnvCfg):
    """Super Detailed: 10 params + adaptive spatial sampling + reward shaping"""
    def __post_init__(self):
        super().__post_init__()
        self.events = SuperDetailedEventCfg()
        self.curriculum = SuperDetailedCurriculumCfg()
