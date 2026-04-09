# Copyright (c) 2024, Curriculum Experiment
# Cartpole with curriculum: pole starts nearly balanced, progressively gets harder.

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

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
# Curriculum parameters
##
_MIN_ANGLE = 0.05                    # Starting range (radians) ~3°
_MAX_ANGLE = 0.25 * math.pi         # Full range ±45° (same as baseline)
_MIN_VEL = 0.05
_MAX_VEL = 0.25 * math.pi
_NUM_STAGES = 10
# Use mean episode length as proxy: if episodes last most of max (300 steps),
# the policy is doing well at the current difficulty level
_PROMOTION_EP_LENGTH = 280           # Out of 300 max steps
_MIN_HOLD_CALLS = 3                  # Min curriculum compute calls before promotion


def curriculum_pole_angle(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
) -> dict[str, float]:
    """Curriculum: progressively widen initial pole angle range.

    Uses mean episode length as the performance signal — if episodes consistently
    last close to the max (300 steps), the policy has mastered the current difficulty
    and we advance to the next stage.
    """

    # Initialize state on first call
    if not hasattr(curriculum_pole_angle, "_state"):
        curriculum_pole_angle._state = {
            "angle_range": _MIN_ANGLE,
            "vel_range": _MIN_VEL,
            "call_count": 0,
            "stage": 0,
        }

    state = curriculum_pole_angle._state
    state["call_count"] += 1

    # env_ids are the environments being reset — use their episode lengths
    if isinstance(env_ids, torch.Tensor):
        reset_ids = env_ids
    elif isinstance(env_ids, slice):
        reset_ids = torch.arange(env.num_envs, device=env.device)
    else:
        reset_ids = torch.tensor(env_ids, device=env.device)

    if reset_ids.numel() > 0:
        # Episode length of environments being reset (these just finished)
        ep_lengths = env.episode_length_buf[reset_ids].float()
        mean_ep_length = ep_lengths.mean().item()

        angle_step = (_MAX_ANGLE - _MIN_ANGLE) / _NUM_STAGES
        vel_step = (_MAX_VEL - _MIN_VEL) / _NUM_STAGES

        if (
            mean_ep_length >= _PROMOTION_EP_LENGTH
            and state["call_count"] >= _MIN_HOLD_CALLS
            and state["stage"] < _NUM_STAGES
        ):
            state["stage"] += 1
            state["angle_range"] = min(_MIN_ANGLE + angle_step * state["stage"], _MAX_ANGLE)
            state["vel_range"] = min(_MIN_VEL + vel_step * state["stage"], _MAX_VEL)
            state["call_count"] = 0

    # Update the reset event parameters in-place
    a = state["angle_range"]
    v = state["vel_range"]
    env.event_manager.cfg.reset_pole_position.params["position_range"] = (-a, a)
    env.event_manager.cfg.reset_pole_position.params["velocity_range"] = (-v, v)

    return {"pole_angle_range": round(a, 4), "pole_vel_range": round(v, 4), "stage": state["stage"]}


##
# Scene
##
@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)))
    robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(prim_path="/World/DomeLight", spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0))

##
# MDP
##
@configclass
class ActionsCfg:
    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=100.0)

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
                "position_range": (-1.0, 1.0), "velocity_range": (-0.5, 0.5)},
    )
    # Start easy: nearly balanced pole
    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
                "position_range": (-0.05, 0.05), "velocity_range": (-0.05, 0.05)},
    )

@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    pole_pos = RewTerm(func=mdp.joint_pos_target_l2, weight=-1.0,
                       params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0})
    cart_vel = RewTerm(func=mdp.joint_vel_l1, weight=-0.01,
                       params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])})
    pole_vel = RewTerm(func=mdp.joint_vel_l1, weight=-0.005,
                       params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])})

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cart_out_of_bounds = DoneTerm(func=mdp.joint_pos_out_of_manual_limit,
                                  params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)})

@configclass
class CurriculumCfg:
    pole_angle_curriculum = CurrTerm(func=curriculum_pole_angle)


##
# Environment config
##
@configclass
class CartpoleCurriculumEnvCfg(ManagerBasedRLEnvCfg):
    scene: CartpoleSceneCfg = CartpoleSceneCfg(num_envs=4096, env_spacing=4.0, clone_in_fabric=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 5
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
