# Copyright (c) 2024, Curriculum Experiment
# Franka Lift Cube with progressive curriculum:
#   Stage 1: Cube spawns near gripper, goal is close and low → learn to grasp
#   Stage 2: Widen cube spawn range → learn to reach + grasp
#   Stage 3: Full random spawn + full goal range → learn complete pick-and-place
#
# The curriculum monitors the "lifting_object" reward as the promotion signal.
# Once the agent consistently lifts the object, we make the task harder.

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import (
    LiftEnvCfg,
    EventCfg,
    RewardsCfg,
    CurriculumCfg as BaseCurriculumCfg,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


##
# Curriculum parameters
##
# Stage definitions: (object_spawn_range, goal_pos_range, goal_z_range)
# Object default position is [0.5, 0, 0.055] on the table
# Default spawn range: x ±0.1, y ±0.25
# Default goal range: x (0.4,0.6), y (-0.25,0.25), z (0.25,0.5)

_STAGES = [
    {  # Stage 0: Very easy — cube near center, goal close and low
        "obj_x": (-0.02, 0.02),
        "obj_y": (-0.05, 0.05),
        "goal_x": (0.45, 0.55),
        "goal_y": (-0.1, 0.1),
        "goal_z": (0.15, 0.25),
    },
    {  # Stage 1: Easy — slightly wider spawn, still easy goal
        "obj_x": (-0.05, 0.05),
        "obj_y": (-0.1, 0.1),
        "goal_x": (0.42, 0.58),
        "goal_y": (-0.15, 0.15),
        "goal_z": (0.2, 0.3),
    },
    {  # Stage 2: Medium — wider spawn, medium goal
        "obj_x": (-0.07, 0.07),
        "obj_y": (-0.15, 0.15),
        "goal_x": (0.4, 0.6),
        "goal_y": (-0.2, 0.2),
        "goal_z": (0.2, 0.4),
    },
    {  # Stage 3: Hard — nearly full range
        "obj_x": (-0.1, 0.1),
        "obj_y": (-0.2, 0.2),
        "goal_x": (0.4, 0.6),
        "goal_y": (-0.25, 0.25),
        "goal_z": (0.25, 0.45),
    },
    {  # Stage 4: Full difficulty — matches baseline
        "obj_x": (-0.1, 0.1),
        "obj_y": (-0.25, 0.25),
        "goal_x": (0.4, 0.6),
        "goal_y": (-0.25, 0.25),
        "goal_z": (0.25, 0.5),
    },
]

_NUM_STAGES = len(_STAGES)
# Promotion: fraction of envs that lifted the object in recent resets
_PROMOTION_LIFT_FRAC = 0.3   # 30% of resetting envs have lifted → promote
_MIN_HOLD_CALLS = 50         # Min curriculum compute calls before promoting
_DEMOTION_ENABLED = False    # Don't go backwards


def lift_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
) -> dict[str, float]:
    """Curriculum: progressively widen object spawn and goal ranges.

    Monitors how many resetting environments had successfully lifted the object
    (object z > minimal_height) during their episode. When enough envs are
    succeeding, we advance to the next difficulty stage.
    """
    # Initialize state
    if not hasattr(lift_curriculum, "_state"):
        lift_curriculum._state = {
            "stage": 0,
            "call_count": 0,
        }

    state = lift_curriculum._state
    state["call_count"] += 1

    # Get resetting env indices
    if isinstance(env_ids, torch.Tensor):
        reset_ids = env_ids
    elif isinstance(env_ids, slice):
        reset_ids = torch.arange(env.num_envs, device=env.device)
    else:
        reset_ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)

    if reset_ids.numel() > 10:  # Need enough envs for a meaningful signal
        # Check how many resetting envs had the object lifted
        # (object z position > table height + some margin)
        from isaaclab.assets import RigidObject
        obj: RigidObject = env.scene["object"]
        obj_heights = obj.data.root_pos_w[reset_ids, 2]
        # The table surface is around z=0.0 in the object's reference
        # Object init is at z=0.055, consider "lifted" if z > 0.1 (above table)
        lifted = (obj_heights > 0.08).float()
        lift_fraction = lifted.mean().item()

        if (
            lift_fraction >= _PROMOTION_LIFT_FRAC
            and state["call_count"] >= _MIN_HOLD_CALLS
            and state["stage"] < _NUM_STAGES - 1
        ):
            state["stage"] += 1
            state["call_count"] = 0

    # Apply current stage settings
    stage_cfg = _STAGES[state["stage"]]

    # Update object spawn range
    env.event_manager.cfg.reset_object_position.params["pose_range"]["x"] = stage_cfg["obj_x"]
    env.event_manager.cfg.reset_object_position.params["pose_range"]["y"] = stage_cfg["obj_y"]

    # Update goal/command range
    cmd_cfg = env.command_manager.cfg.object_pose
    cmd_cfg.ranges.pos_x = stage_cfg["goal_x"]
    cmd_cfg.ranges.pos_y = stage_cfg["goal_y"]
    cmd_cfg.ranges.pos_z = stage_cfg["goal_z"]

    return {
        "stage": state["stage"],
        "obj_x_range": stage_cfg["obj_x"][1],
        "goal_z_max": stage_cfg["goal_z"][1],
    }


##
# Curriculum config
##
@configclass
class LiftCurriculumCfg:
    """Enhanced curriculum for lift task — progressive difficulty."""

    # Keep the original reward weight curriculum
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000},
    )
    # Add our progressive difficulty curriculum
    lift_progressive = CurrTerm(func=lift_curriculum)


##
# Custom event config — starts with easy spawn range
##
from isaaclab.managers import EventTermCfg as EventTerm

@configclass
class LiftCurriculumEventCfg:
    """Events for curriculum version — starts with tight object spawn."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.02, 0.02), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )
