# LLM-Crafted Curriculum v2 for Dexsuite Kuka-Allegro Lift
#
# KEY IMPROVEMENTS OVER v1:
#   1. Per-env difficulty (0.0→1.0) — each env adjusts independently
#   2. Smooth interpolation of ALL parameters — no discrete jumps
#   3. LLM-designed parameter schedules — each parameter has a custom curve
#   4. Demotion — difficulty decreases on failure
#
# The LLM's contribution: designing the SHAPE of each parameter's schedule.
# Instead of ADR's linear interpolation for everything, each parameter has
# a custom curve based on task analysis:
#   - Gravity ramps SLOWLY early (grasping is the bottleneck)
#   - Spawn range widens FASTER (reaching is relatively easy)
#   - Goal range expands MEDIUM (needs some grasping skill first)
#   - Joint noise ramps LATE (robustness after skill acquisition)
#   - Rotation range ramps MEDIUM (orientation diversity)

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
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ──────────────────────────────────────────────────────────────────────────────
# LLM-designed parameter schedules
# Each function maps difficulty ∈ [0,1] → parameter value
# The SHAPE of these curves is the LLM's key contribution
# ──────────────────────────────────────────────────────────────────────────────

def _gravity_schedule(d: float) -> float:
    """Gravity ramps SLOWLY early, then accelerates.
    Rationale: grasping in zero-G is the critical first skill. Don't add
    gravity until the hand has basic finger coordination.
    Curve: cubic — stays near 0 for d<0.3, then ramps steeply."""
    return -9.81 * (d ** 1.8)


def _obj_spawn_x_schedule(d: float) -> tuple[float, float]:
    """Object x spawn range widens faster than gravity.
    Rationale: reaching is easier than grasping, so widen spawn early."""
    half = 0.02 + 0.18 * (d ** 1.2)  # 0.02 → 0.20
    return (-half, half)


def _obj_spawn_y_schedule(d: float) -> tuple[float, float]:
    """Same as x but slightly different range."""
    half = 0.02 + 0.18 * (d ** 1.2)
    return (-half, half)


def _obj_spawn_z_schedule(d: float) -> tuple[float, float]:
    """Object z spawn — height above table. Ramps linearly."""
    return (0.0, 0.4 * d)


def _obj_rotation_schedule(d: float) -> float:
    """Object rotation range. Ramps at medium pace."""
    return 0.5 + 2.64 * (d ** 1.5)  # 0.5 → 3.14 rad


def _goal_x_schedule(d: float) -> tuple[float, float]:
    """Goal x range widens at medium pace."""
    center = -0.55
    half = 0.05 + 0.15 * (d ** 1.3)  # ±0.05 → ±0.20 around center
    return (center - half, center + half)


def _goal_y_schedule(d: float) -> tuple[float, float]:
    """Goal y range."""
    half = 0.05 + 0.20 * (d ** 1.3)  # ±0.05 → ±0.25
    return (-half, half)


def _goal_z_schedule(d: float) -> tuple[float, float]:
    """Goal z — height target. Starts close to table, expands up."""
    lo = 0.35 + 0.20 * d       # 0.35 → 0.55
    hi = 0.45 + 0.50 * d       # 0.45 → 0.95
    return (lo, hi)


def _joint_reset_range_schedule(d: float) -> float:
    """Joint randomization range. Ramps linearly."""
    return 0.1 + 0.4 * d  # 0.1 → 0.5 rad


def _obs_noise_schedule(d: float) -> float:
    """Observation noise. Ramps LATE — want clean signal for skill learning.
    Curve: stays near 0 for d<0.5, then ramps."""
    return d ** 3.0  # very late ramp


# ──────────────────────────────────────────────────────────────────────────────
# Per-env difficulty tracker
# ──────────────────────────────────────────────────────────────────────────────
class PerEnvDifficulty:
    """Tracks per-environment difficulty level (0.0 to 1.0).
    
    On reset: checks if the env succeeded (object near goal).
    If yes: difficulty += step_up
    If no: difficulty -= step_down (demotion)
    """

    def __init__(self, num_envs: int, device: torch.device,
                 step_up: float = 0.02, step_down: float = 0.01,
                 min_d: float = 0.0, max_d: float = 1.0,
                 promotion_only_until: float = 0.5):
        self.difficulties = torch.zeros(num_envs, device=device)
        self.step_up = step_up
        self.step_down = step_down
        self.min_d = min_d
        self.max_d = max_d
        self.promotion_only_until = promotion_only_until
        self.device = device

    def update(self, env_ids: torch.Tensor, success: torch.Tensor):
        """Update difficulty for resetting envs based on success.
        
        Promotion-only until d reaches promotion_only_until (default 0.5).
        After that, demotion is enabled for fine-tuning.
        """
        if env_ids.numel() == 0:
            return
        d = self.difficulties[env_ids]
        # Promotion-only phase: no demotion until env reaches threshold
        promotion_only_mask = d < self.promotion_only_until
        step_down = torch.where(promotion_only_mask,
                                torch.zeros_like(d),
                                torch.full_like(d, self.step_down))
        delta = torch.where(success, self.step_up, -step_down)
        self.difficulties[env_ids] = (d + delta).clamp(self.min_d, self.max_d)

    @property
    def mean_difficulty(self) -> float:
        return self.difficulties.mean().item()


# ──────────────────────────────────────────────────────────────────────────────
# Smooth per-env reset — applies difficulty-dependent parameters per env
# ──────────────────────────────────────────────────────────────────────────────
def smooth_reset_object(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Reset object with per-env difficulty-dependent spawn parameters.
    Each env gets a spawn range determined by its own difficulty level."""
    asset: RigidObject = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()
    n = len(env_ids)

    tracker = getattr(llm_curriculum_v2, "_tracker", None)
    if tracker is None:
        # Fallback: uniform
        range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (n, 6), device=asset.device)
    else:
        d = tracker.difficulties[env_ids]  # (n,) per-env difficulty
        rand_samples = torch.zeros(n, 6, device=asset.device)

        # Per-env spawn ranges based on difficulty
        for i in range(n):
            di = d[i].item()
            x_lo, x_hi = _obj_spawn_x_schedule(di)
            y_lo, y_hi = _obj_spawn_y_schedule(di)
            z_lo, z_hi = _obj_spawn_z_schedule(di)
            rot_range = _obj_rotation_schedule(di)

            rand_samples[i, 0] = torch.empty(1, device=asset.device).uniform_(x_lo, x_hi)
            rand_samples[i, 1] = torch.empty(1, device=asset.device).uniform_(y_lo, y_hi)
            rand_samples[i, 2] = torch.empty(1, device=asset.device).uniform_(z_lo, z_hi)
            rand_samples[i, 3] = torch.empty(1, device=asset.device).uniform_(-rot_range, rot_range)
            rand_samples[i, 4] = torch.empty(1, device=asset.device).uniform_(-rot_range, rot_range)
            rand_samples[i, 5] = torch.empty(1, device=asset.device).uniform_(-rot_range, rot_range)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    vel_range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    vel_ranges = torch.tensor(vel_range_list, device=asset.device)
    vel_samples = math_utils.sample_uniform(vel_ranges[:, 0], vel_ranges[:, 1], (n, 6), device=asset.device)
    velocities = root_states[:, 7:13] + vel_samples

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def smooth_reset_joints(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    position_range: list[float],
    velocity_range: list[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset joints with per-env difficulty-dependent noise range."""
    asset: Articulation = env.scene[asset_cfg.name]

    tracker = getattr(llm_curriculum_v2, "_tracker", None)
    if tracker is None:
        joint_pos = asset.data.default_joint_pos[env_ids].clone()
        joint_vel = asset.data.default_joint_vel[env_ids].clone()
        noise = math_utils.sample_uniform(position_range[0], position_range[1],
                                          joint_pos.shape, joint_pos.device)
        joint_pos += noise
    else:
        joint_pos = asset.data.default_joint_pos[env_ids].clone()
        joint_vel = asset.data.default_joint_vel[env_ids].clone()
        d = tracker.difficulties[env_ids]
        for i in range(len(env_ids)):
            r = _joint_reset_range_schedule(d[i].item())
            noise = torch.empty_like(joint_pos[i]).uniform_(-r, r)
            joint_pos[i] += noise

    joint_pos = joint_pos.clamp_(asset.data.joint_pos_limits[env_ids, :, 0],
                                  asset.data.joint_pos_limits[env_ids, :, 1])
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


# ──────────────────────────────────────────────────────────────────────────────
# Main curriculum function
# ──────────────────────────────────────────────────────────────────────────────
def llm_curriculum_v2(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
) -> dict[str, float]:
    """LLM Curriculum v2: per-env smooth interpolation of all parameters.
    
    Each env has its own difficulty (0→1). On reset:
    - Check if object was near goal → promote or demote
    - Apply gravity based on MEAN difficulty (physics-wide setting)
    - Spawn range, goal range, joint noise are per-env (set by reset functions)
    """
    if not hasattr(llm_curriculum_v2, "_tracker"):
        llm_curriculum_v2._tracker = PerEnvDifficulty(
            num_envs=env.num_envs,
            device=env.device,
            step_up=0.015,    # gradual promotion
            step_down=0.008,  # slower demotion
        )
        llm_curriculum_v2._call_count = 0

    tracker = llm_curriculum_v2._tracker
    llm_curriculum_v2._call_count += 1

    # Resolve env_ids
    if isinstance(env_ids, torch.Tensor):
        reset_ids = env_ids
    elif isinstance(env_ids, slice):
        reset_ids = torch.arange(env.num_envs, device=env.device)
    else:
        reset_ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)

    # Update difficulty based on outcomes
    if reset_ids.numel() > 0 and llm_curriculum_v2._call_count > 2:
        robot: Articulation = env.scene["robot"]
        obj: RigidObject = env.scene["object"]
        command = env.command_manager.get_command("object_pose")
        des_pos_w, _ = combine_frame_transforms(
            robot.data.root_pos_w[reset_ids], robot.data.root_quat_w[reset_ids],
            command[reset_ids, :3], command[reset_ids, 3:7]
        )
        pos_dist = torch.norm(des_pos_w - obj.data.root_pos_w[reset_ids], dim=1)
        success = pos_dist < 0.15  # within 15cm = success
        tracker.update(reset_ids, success)

    # Apply GRAVITY based on mean difficulty (can't be per-env, it's scene-wide)
    mean_d = tracker.mean_difficulty
    gz = _gravity_schedule(mean_d)
    env.event_manager.cfg.variable_gravity.params["gravity_distribution_params"] = (
        (0.0, 0.0, gz), (0.0, 0.0, gz)
    )

    # Apply GOAL RANGE based on mean difficulty (command manager is global)
    gx = _goal_x_schedule(mean_d)
    gy = _goal_y_schedule(mean_d)
    gz_goal = _goal_z_schedule(mean_d)
    cmd_cfg = env.command_manager.cfg.object_pose
    cmd_cfg.ranges.pos_x = gx
    cmd_cfg.ranges.pos_y = gy
    cmd_cfg.ranges.pos_z = gz_goal

    # Object spawn and joint noise are per-env (handled by custom reset functions)

    # Compute percentile stats
    d = tracker.difficulties
    d10 = torch.quantile(d, 0.1).item()
    d50 = torch.quantile(d, 0.5).item()
    d90 = torch.quantile(d, 0.9).item()

    return {
        "mean_difficulty": mean_d,
        "d10": d10,
        "d50_median": d50,
        "d90": d90,
        "gravity_z": gz,
        "goal_z_max": gz_goal[1],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Config classes
# ──────────────────────────────────────────────────────────────────────────────
@configclass
class LLMv2CurriculumCfg:
    """LLM Curriculum v2 — per-env smooth interpolation, no ADR."""
    llm_v2 = CurrTerm(func=llm_curriculum_v2)


@configclass
class LLMv2EventCfg(DexsuiteEventCfg):
    """Events with smooth per-env difficulty-dependent resets."""

    reset_object = EventTerm(
        func=smooth_reset_object,
        mode="reset",
        params={
            "pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (0.0, 0.0),
                           "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    variable_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="reset",
        params={
            "gravity_distribution_params": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            "operation": "abs",
        },
    )

    reset_robot_joints = EventTerm(
        func=smooth_reset_joints,
        mode="reset",
        params={
            "position_range": [-0.1, 0.1],
            "velocity_range": [0.0, 0.0],
        },
    )
