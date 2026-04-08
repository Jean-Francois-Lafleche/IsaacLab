# LLM-Crafted Progressive Curriculum for Dexsuite Kuka-Allegro Lift
# 
# DESIGN RATIONALE (by LLM task analysis):
#
# The Dexsuite lift task requires 6 sub-skills learned in sequence:
#   1. Finger coordination — basic motor control without breaking physics
#   2. Reaching — move arm/hand near the object
#   3. Grasping — close fingers with proper multi-finger contact
#   4. Holding — maintain grasp against increasing gravity
#   5. Lifting — raise object off table toward goal height
#   6. Placing — transport and position object at goal
#
# Unlike ADR which uses a single scalar difficulty and interpolates everything,
# this curriculum has EXPLICIT STAGES with task-specific parameter schedules:
#
# Stage 0: REACH (gravity=0, object right in front of hand, no goal tracking needed)
#   → Learn arm + finger motor control, reach toward object
#
# Stage 1: GRASP (gravity=0, object close, small goal near object) 
#   → Learn multi-finger contact, boost contact reward weight
#
# Stage 2: HOLD (gravity ramps 0→3 m/s², object close, goal near object)
#   → Learn to maintain grasp against light gravity
#
# Stage 3: LIFT (gravity ramps 3→7 m/s², wider spawn, goal above table)
#   → Learn to counteract stronger gravity, lift object
#
# Stage 4: TRANSPORT (gravity=7→9.81, full spawn range, wider goals)
#   → Learn to move object to distant goals under full gravity
#
# Stage 5: MASTER (full gravity, full randomization, full goal range)
#   → Polish performance at full difficulty = matches baseline
#
# PROMOTION: Based on *task-specific metrics* per stage, not generic difficulty.
# Each stage has a custom promotion condition tied to the sub-skill it teaches.

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import combine_frame_transforms

from isaaclab_tasks.manager_based.manipulation.dexsuite import mdp
from isaaclab_tasks.manager_based.manipulation.dexsuite.dexsuite_env_cfg import (
    EventCfg as DexsuiteEventCfg,
    CommandsCfg,
    RewardsCfg,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ──────────────────────────────────────────────────────────────────────────────
# Stage definitions — each stage teaches a specific sub-skill
# ──────────────────────────────────────────────────────────────────────────────
_STAGES = {
    0: {  # REACH — learn arm/finger motor control
        "name": "reach",
        "gravity": (0.0, 0.0, 0.0),
        "obj_pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0),
                           "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5)},
        "goal_range": {"pos_x": (-0.6, -0.5), "pos_y": (-0.05, 0.05), "pos_z": (0.35, 0.45)},
        "joint_reset_range": (-0.2, 0.2),  # smaller joint randomization
        "promote_metric": "fingers_to_object",
        "promote_threshold": 0.22,  # finger-to-object reward > 0.22 (close proximity)
        "min_hold_iters": 30,
    },
    1: {  # GRASP — learn multi-finger contact
        "name": "grasp",
        "gravity": (0.0, 0.0, 0.0),
        "obj_pose_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08), "z": (0.0, 0.05),
                           "roll": (-1.0, 1.0), "pitch": (-1.0, 1.0), "yaw": (-1.0, 1.0)},
        "goal_range": {"pos_x": (-0.65, -0.45), "pos_y": (-0.1, 0.1), "pos_z": (0.4, 0.55)},
        "joint_reset_range": (-0.3, 0.3),
        "promote_metric": "success",  # success reward requires object near goal
        "promote_threshold": 0.02,
        "min_hold_iters": 40,
    },
    2: {  # HOLD — maintain grasp under light gravity
        "name": "hold",
        "gravity": (0.0, 0.0, -3.0),
        "obj_pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.0, 0.1),
                           "roll": (-1.5, 1.5), "pitch": (-1.5, 1.5), "yaw": (-1.5, 1.5)},
        "goal_range": {"pos_x": (-0.65, -0.4), "pos_y": (-0.15, 0.15), "pos_z": (0.45, 0.65)},
        "joint_reset_range": (-0.4, 0.4),
        "promote_metric": "success",
        "promote_threshold": 0.02,
        "min_hold_iters": 50,
    },
    3: {  # LIFT — counteract stronger gravity, lift off table
        "name": "lift",
        "gravity": (0.0, 0.0, -6.0),
        "obj_pose_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15), "z": (0.0, 0.2),
                           "roll": (-2.5, 2.5), "pitch": (-2.5, 2.5), "yaw": (-2.5, 2.5)},
        "goal_range": {"pos_x": (-0.7, -0.35), "pos_y": (-0.2, 0.2), "pos_z": (0.5, 0.8)},
        "joint_reset_range": (-0.45, 0.45),
        "promote_metric": "success",
        "promote_threshold": 0.015,
        "min_hold_iters": 60,
    },
    4: {  # TRANSPORT — full gravity, wider goals
        "name": "transport",
        "gravity": (0.0, 0.0, -9.0),
        "obj_pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (0.0, 0.3),
                           "roll": (-3.14, 3.14), "pitch": (-3.14, 3.14), "yaw": (-3.14, 3.14)},
        "goal_range": {"pos_x": (-0.7, -0.3), "pos_y": (-0.25, 0.25), "pos_z": (0.55, 0.9)},
        "joint_reset_range": (-0.5, 0.5),
        "promote_metric": "success",
        "promote_threshold": 0.01,
        "min_hold_iters": 60,
    },
    5: {  # MASTER — full difficulty, matches baseline
        "name": "master",
        "gravity": (0.0, 0.0, -9.81),
        "obj_pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (0.0, 0.4),
                           "roll": (-3.14, 3.14), "pitch": (-3.14, 3.14), "yaw": (-3.14, 3.14)},
        "goal_range": {"pos_x": (-0.7, -0.3), "pos_y": (-0.25, 0.25), "pos_z": (0.55, 0.95)},
        "joint_reset_range": (-0.5, 0.5),
        "promote_metric": None,  # terminal stage — no promotion
        "promote_threshold": None,
        "min_hold_iters": None,
    },
}

_NUM_STAGES = len(_STAGES)


# ──────────────────────────────────────────────────────────────────────────────
# LLM Curriculum — main curriculum term
# ──────────────────────────────────────────────────────────────────────────────
def llm_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
) -> dict[str, float]:
    """LLM-designed progressive curriculum for Dexsuite Kuka-Allegro Lift.
    
    Unlike ADR (which uses a single difficulty scalar), this curriculum has
    explicit stages with task-specific promotion conditions:
    - Stage 0 (REACH): promotes when fingers are close to object
    - Stage 1 (GRASP): promotes when object reaches goal (in zero-G)  
    - Stage 2 (HOLD): promotes when object reaches goal under light gravity
    - Stage 3 (LIFT): promotes when object reaches goal under medium gravity
    - Stage 4 (TRANSPORT): promotes when object reaches goal under strong gravity
    - Stage 5 (MASTER): terminal — full difficulty
    """
    # ── Initialize state ──
    if not hasattr(llm_curriculum, "_state"):
        llm_curriculum._state = {
            "stage": 0,
            "call_count": 0,
            "stage_call_count": 0,
        }

    state = llm_curriculum._state
    state["call_count"] += 1
    state["stage_call_count"] += 1

    # ── Resolve env_ids ──
    if isinstance(env_ids, torch.Tensor):
        reset_ids = env_ids
    elif isinstance(env_ids, slice):
        reset_ids = torch.arange(env.num_envs, device=env.device)
    else:
        reset_ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)

    current_stage = _STAGES[state["stage"]]

    # ── Check promotion condition ──
    if (current_stage["promote_metric"] is not None
        and reset_ids.numel() > 20
        and state["stage_call_count"] >= current_stage["min_hold_iters"]
        and state["stage"] < _NUM_STAGES - 1):

        metric = current_stage["promote_metric"]
        threshold = current_stage["promote_threshold"]

        if metric == "fingers_to_object":
            # Check reaching reward
            robot: Articulation = env.scene["robot"]
            obj: RigidObject = env.scene["object"]
            asset_pos = robot.data.body_pos_w[reset_ids]
            # Use palm + fingertip bodies
            body_ids = [i for i, n in enumerate(robot.body_names) if "tip" in n or "palm" in n]
            if body_ids:
                ee_pos = asset_pos[:, body_ids]
                obj_pos = obj.data.root_pos_w[reset_ids].unsqueeze(1)
                dists = torch.norm(ee_pos - obj_pos, dim=-1).min(dim=-1).values
                avg_dist = dists.mean().item()
                # fingers_to_object reward = 1 - tanh(dist/0.4)
                reward_proxy = 1.0 - math.tanh(avg_dist / 0.4)
                if reward_proxy > threshold:
                    state["stage"] += 1
                    state["stage_call_count"] = 0

        elif metric == "success":
            # Check if object is near goal
            robot: Articulation = env.scene["robot"]
            obj: RigidObject = env.scene["object"]
            command = env.command_manager.get_command("object_pose")
            des_pos_w, _ = combine_frame_transforms(
                robot.data.root_pos_w[reset_ids], robot.data.root_quat_w[reset_ids],
                command[reset_ids, :3], command[reset_ids, 3:7]
            )
            pos_dist = torch.norm(des_pos_w - obj.data.root_pos_w[reset_ids], dim=1)
            success_frac = (pos_dist < 0.15).float().mean().item()
            # success reward proxy = (1 - tanh(dist/0.1))^2
            avg_success = ((1 - torch.tanh(pos_dist / 0.1)) ** 2).mean().item()
            if avg_success > threshold:
                state["stage"] += 1
                state["stage_call_count"] = 0

    # ── Apply current stage settings ──
    stage_cfg = _STAGES[state["stage"]]

    # Update gravity
    gx, gy, gz = stage_cfg["gravity"]
    env.event_manager.cfg.variable_gravity.params["gravity_distribution_params"] = (
        (gx, gy, gz), (gx, gy, gz)
    )

    # Update object spawn range
    env.event_manager.cfg.reset_object.params["pose_range"] = stage_cfg["obj_pose_range"]

    # Update joint reset range
    env.event_manager.cfg.reset_robot_joints.params["position_range"] = list(stage_cfg["joint_reset_range"])

    # Update goal/command range
    cmd_cfg = env.command_manager.cfg.object_pose
    for key, val in stage_cfg["goal_range"].items():
        setattr(cmd_cfg.ranges, key, val)

    return {
        "stage": state["stage"],
        "stage_name": float(state["stage"]),  # tensorboard only takes floats
        "gravity_z": stage_cfg["gravity"][2],
        "call_count": state["call_count"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Config classes
# ──────────────────────────────────────────────────────────────────────────────
@configclass
class LLMCurriculumCfg:
    """LLM-crafted progressive curriculum — no ADR, pure task-specific stages."""
    llm_progressive = CurrTerm(func=llm_curriculum)


@configclass
class LLMCurriculumEventCfg(DexsuiteEventCfg):
    """Events for LLM curriculum — starts with easy settings.
    
    Override object spawn to start with tight range (Stage 0: REACH).
    The curriculum function will modify these dynamically.
    """
    # Override object reset with tight initial range
    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (0.0, 0.0),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
            "velocity_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (-0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    # Override gravity to start at zero
    variable_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="reset",
        params={
            "gravity_distribution_params": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            "operation": "abs",
        },
    )

    # Tighter joint reset for Stage 0
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": [-0.2, 0.2],
            "velocity_range": [0.0, 0.0],
        },
    )
