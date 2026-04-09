# Copyright (c) 2024, Stack RL Experiment
# RL environment config for Franka cube stacking with PPO rewards.

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_joint_pos_env_cfg import (
    FrankaCubeStackEnvCfg,
)
from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import rewards as stack_rewards


@configclass
class RewardsCfg:
    """Minimal rewards — let the curriculum do the heavy lifting."""

    reaching_cube1 = RewTerm(
        func=stack_rewards.reaching_cube1,
        params={"std": 0.1},
        weight=1.0,
    )

    stacking_success = RewTerm(
        func=stack_rewards.stacking_success,
        weight=10.0,
    )

    action_rate = RewTerm(
        func=stack_rewards.action_rate,
        weight=-0.5,
    )


@configclass
class RLObservationsCfg:
    """RL-only observations — single concatenated policy group."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object = ObsTerm(func=mdp.object_obs)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class FrankaStackRLEnvCfg(FrankaCubeStackEnvCfg):
    """Franka cube stack with RL rewards (baseline — no curriculum)."""

    def __post_init__(self):
        super().__post_init__()

        # RL-specific timing: match the lift task pattern
        self.decimation = 2
        self.episode_length_s = 10.0
        self.sim.dt = 1.0 / 120.0  # 120 Hz
        self.sim.render_interval = self.decimation

        # Enable rewards for PPO training
        self.rewards = RewardsCfg()

        # RL-only observations: single concatenated policy group (no subtask/rgb groups)
        self.observations = RLObservationsCfg()

        # Disable XR config (contains lambdas that break Hydra serialization)
        self.xr = None


@configclass
class FrankaStackRLEnvCfg_PLAY(FrankaStackRLEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
