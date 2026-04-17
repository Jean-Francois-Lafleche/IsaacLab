# Copyright (c) 2024, Curriculum Experiment
# Cartpole with curriculum: pole starts nearly balanced, progressively gets harder.
#
# The curriculum tracks mean episode reward. As it improves, the initial pole angle
# range widens from near-zero to the full ±45° (0.25*pi) used in the standard task.

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.cartpole.mdp as mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip


##
# Curriculum function
##

# Global state for the curriculum — tracks the current difficulty level
_curriculum_state = {
    "pole_angle_range": 0.05,  # Start with ±0.05 rad (~3°) — nearly balanced
    "pole_vel_range": 0.05,    # Start with very small angular velocity
    "step_count": 0,
}

# Curriculum parameters
_MIN_ANGLE = 0.05                    # Starting range (radians) — nearly balanced
_MAX_ANGLE = 0.25 * math.pi         # Full range (same as baseline: ±45°)
_MIN_VEL = 0.05                      # Starting velocity range
_MAX_VEL = 0.25 * math.pi           # Full velocity range (same as baseline)
_NUM_CURRICULUM_STAGES = 10          # Number of difficulty stages
_PROMOTION_REWARD_THRESHOLD = 3.5    # Mean reward threshold to advance (max possible ~5.0)
_STAGE_HOLD_STEPS = 5               # Min curriculum updates before promotion


def curriculum_pole_angle(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
) -> dict[str, float]:
    """Curriculum term: progressively increase initial pole angle range.

    Monitors the mean episode reward across all environments. When it exceeds
    the threshold, advances to the next difficulty stage (wider initial angles).
    """
    _curriculum_state["step_count"] += 1

    # Get current mean reward from the reward manager
    # We use the total episode reward tracked by the environment
    if hasattr(env, "episode_length_buf") and hasattr(env, "reward_buf"):
        # Check environments that just terminated to gauge performance
        terminated = env.termination_manager.terminated | env.termination_manager.time_outs
        if terminated.any():
            # Mean reward per step for terminated episodes
            term_indices = terminated.nonzero(as_tuple=False).squeeze(-1)
            if term_indices.numel() > 0:
                ep_lengths = env.episode_length_buf[term_indices].float()
                # Use current reward buffer as proxy for recent performance
                mean_reward = env.reward_buf.mean().item()

                # Determine current stage
                current_angle = _curriculum_state["pole_angle_range"]
                stage_size = (_MAX_ANGLE - _MIN_ANGLE) / _NUM_CURRICULUM_STAGES

                if (
                    mean_reward > _PROMOTION_REWARD_THRESHOLD
                    and _curriculum_state["step_count"] > _STAGE_HOLD_STEPS
                    and current_angle < _MAX_ANGLE
                ):
                    # Promote to next stage
                    _curriculum_state["pole_angle_range"] = min(
                        current_angle + stage_size, _MAX_ANGLE
                    )
                    _curriculum_state["pole_vel_range"] = min(
                        _curriculum_state["pole_vel_range"]
                        + (_MAX_VEL - _MIN_VEL) / _NUM_CURRICULUM_STAGES,
                        _MAX_VEL,
                    )
                    _curriculum_state["step_count"] = 0

    # Update the event config to use the new ranges
    angle_range = _curriculum_state["pole_angle_range"]
    vel_range = _curriculum_state["pole_vel_range"]

    # Modify the reset event parameters in-place
    env.event_manager.cfg.reset_pole_position.params["position_range"] = (-angle_range, angle_range)
    env.event_manager.cfg.reset_pole_position.params["velocity_range"] = (-vel_range, vel_range)

    return {
        "pole_angle_range": angle_range,
        "pole_vel_range": vel_range,
    }


##
# Scene (same as standard)
##

@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##

@configclass
class ActionsCfg:
    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot", joint_names=["slider_to_cart"], scale=100.0
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events — initial ranges are small (curriculum starts easy)."""
    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.5, 0.5),
        },
    )
    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            # Start easy: nearly balanced
            "position_range": (-0.05, 0.05),
            "velocity_range": (-0.05, 0.05),
        },
    )


@configclass
class RewardsCfg:
    """Same rewards as standard cartpole."""
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    pole_pos = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
    )
    cart_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    )
    pole_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
    )


@configclass
class CurriculumCfg:
    """Curriculum configuration — progressively widen initial pole angles."""
    pole_angle_curriculum = CurrTerm(func=curriculum_pole_angle)


##
# Full environment config
##

@configclass
class CartpoleCurriculumEnvCfg(ManagerBasedRLEnvCfg):
    """Cartpole with curriculum — starts easy, progressively gets harder."""
    scene: CartpoleSceneCfg = CartpoleSceneCfg(num_envs=4096, env_spacing=4.0, clone_in_fabric=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 5
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation


##
# Register the curriculum environment
##

gym.register(
    id="Isaac-Cartpole-Curriculum-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:CartpoleCurriculumEnvCfg",
    },
)
