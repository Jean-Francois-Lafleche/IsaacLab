# LLM-Crafted Curriculum v3 for Dexsuite Kuka-Allegro Lift
#
# v3 = v2 (per-env smooth interpolation) + adaptive failure-biased sampling
#
# New in v3:
#   5. Failure-biased spatial sampling — track success per spatial bin,
#      over-sample from bins where the policy fails most

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

# Import v2's schedules and PerEnvDifficulty
from isaaclab_tasks.manager_based.manipulation.dexsuite.llm_curriculum_v2 import (
    _gravity_schedule,
    _obj_spawn_x_schedule,
    _obj_spawn_y_schedule,
    _obj_spawn_z_schedule,
    _obj_rotation_schedule,
    _goal_x_schedule,
    _goal_y_schedule,
    _goal_z_schedule,
    _joint_reset_range_schedule,
    _obs_noise_schedule,
    PerEnvDifficulty,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ──────────────────────────────────────────────────────────────────────────────
# Spatial failure tracker — bins initial states, tracks success per bin
# ──────────────────────────────────────────────────────────────────────────────
class SpatialFailureTracker:
    """Track success rates per spatial bin of object spawn position.
    
    Bins the (x, y, z) offset space into a grid and maintains an EMA
    of success rate per bin. Returns sampling weights inversely
    proportional to success (harder bins get sampled more).
    """

    def __init__(self, device: torch.device, num_envs: int,
                 x_bins: int = 4, y_bins: int = 4, z_bins: int = 4,
                 ema_alpha: float = 0.05):
        self.device = device
        self.num_envs = num_envs
        self.x_bins = x_bins
        self.y_bins = y_bins
        self.z_bins = z_bins
        self.total_bins = x_bins * y_bins * z_bins
        self.ema_alpha = ema_alpha

        # EMA of success rate per bin (0.5 = uncertain)
        self.success_ema = torch.full((self.total_bins,), 0.5, device=device)
        self.bin_counts = torch.zeros(self.total_bins, device=device)
        # Per-env: which bin this env's episode started in
        self.env_bins = torch.zeros(num_envs, dtype=torch.long, device=device)

    def get_bin_idx(self, x_frac: torch.Tensor, y_frac: torch.Tensor,
                    z_frac: torch.Tensor) -> torch.Tensor:
        """Convert normalized [0,1] positions to flat bin indices."""
        ix = (x_frac * self.x_bins).long().clamp(0, self.x_bins - 1)
        iy = (y_frac * self.y_bins).long().clamp(0, self.y_bins - 1)
        iz = (z_frac * self.z_bins).long().clamp(0, self.z_bins - 1)
        return ix * (self.y_bins * self.z_bins) + iy * self.z_bins + iz

    def record_reset(self, env_ids: torch.Tensor, bin_indices: torch.Tensor):
        """Record which bin each env started in."""
        self.env_bins[env_ids] = bin_indices

    def record_outcomes(self, env_ids: torch.Tensor, success: torch.Tensor):
        """Update success EMA for bins these envs came from."""
        if env_ids.numel() == 0:
            return
        bins = self.env_bins[env_ids]
        s = success.float()
        for i in range(env_ids.numel()):
            b = bins[i].item()
            self.success_ema[b] = (
                (1 - self.ema_alpha) * self.success_ema[b] + self.ema_alpha * s[i]
            )
            self.bin_counts[b] += 1

    def get_sampling_weights(self) -> torch.Tensor:
        """Weights ∝ failure rate + exploration bonus. Higher = harder bin."""
        failure_rate = 1.0 - self.success_ema
        confidence = (self.bin_counts / (self.bin_counts.max() + 1)).clamp(0, 1)
        exploration_bonus = 0.3 * (1.0 - confidence)
        weights = (failure_rate + exploration_bonus).clamp(min=0.05)
        return weights / weights.sum()


# ──────────────────────────────────────────────────────────────────────────────
# Failure-biased per-env reset — combines difficulty curves + spatial sampling
# ──────────────────────────────────────────────────────────────────────────────
def adaptive_smooth_reset_object(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Reset object with per-env difficulty AND failure-biased spatial sampling."""
    asset: RigidObject = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()
    n = len(env_ids)

    state = getattr(llm_curriculum_v3, "_state", None)
    if state is None or state.get("warmup", True):
        # Fallback: uniform (same as v2)
        range_list = [pose_range.get(key, (0.0, 0.0))
                      for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (n, 6), device=asset.device)
    else:
        tracker = state["difficulty"]
        failure = state["failure"]
        d = tracker.difficulties[env_ids]

        rand_samples = torch.zeros(n, 6, device=asset.device)

        # Get failure-biased sampling weights
        weights = failure.get_sampling_weights()
        # Sample target bins proportional to failure
        bin_indices = torch.multinomial(weights, n, replacement=True)

        # Decompose flat bin → (ix, iy, iz)
        iz = bin_indices % failure.z_bins
        remainder = bin_indices // failure.z_bins
        iy = remainder % failure.y_bins
        ix = remainder // failure.y_bins

        for i in range(n):
            di = d[i].item()
            # Get the full range for this env's difficulty
            x_lo, x_hi = _obj_spawn_x_schedule(di)
            y_lo, y_hi = _obj_spawn_y_schedule(di)
            z_lo, z_hi = _obj_spawn_z_schedule(di)
            rot_range = _obj_rotation_schedule(di)

            x_range = x_hi - x_lo
            y_range = y_hi - y_lo
            z_range = max(z_hi - z_lo, 1e-6)

            # Sample within the TARGET BIN (failure-biased)
            bw_x = x_range / failure.x_bins
            bw_y = y_range / failure.y_bins
            bw_z = z_range / failure.z_bins

            # Position within the chosen bin
            obj_x = x_lo + ix[i].item() * bw_x + torch.rand(1, device=asset.device).item() * bw_x
            obj_y = y_lo + iy[i].item() * bw_y + torch.rand(1, device=asset.device).item() * bw_y
            obj_z = z_lo + iz[i].item() * bw_z + torch.rand(1, device=asset.device).item() * bw_z

            rand_samples[i, 0] = obj_x
            rand_samples[i, 1] = obj_y
            rand_samples[i, 2] = obj_z
            rand_samples[i, 3] = torch.empty(1, device=asset.device).uniform_(-rot_range, rot_range)
            rand_samples[i, 4] = torch.empty(1, device=asset.device).uniform_(-rot_range, rot_range)
            rand_samples[i, 5] = torch.empty(1, device=asset.device).uniform_(-rot_range, rot_range)

        # Compute normalized fractions for bin tracking
        # Use the mean difficulty's range as reference for binning
        mean_d = tracker.mean_difficulty
        mx_lo, mx_hi = _obj_spawn_x_schedule(mean_d)
        my_lo, my_hi = _obj_spawn_y_schedule(mean_d)
        mz_lo, mz_hi = _obj_spawn_z_schedule(mean_d)

        x_frac = ((rand_samples[:, 0] - mx_lo) / max(mx_hi - mx_lo, 1e-6)).clamp(0, 0.999)
        y_frac = ((rand_samples[:, 1] - my_lo) / max(my_hi - my_lo, 1e-6)).clamp(0, 0.999)
        z_frac = ((rand_samples[:, 2] - mz_lo) / max(mz_hi - mz_lo, 1e-6)).clamp(0, 0.999)
        actual_bins = failure.get_bin_idx(x_frac, y_frac, z_frac)
        failure.record_reset(env_ids, actual_bins)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(
        rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    vel_range_list = [velocity_range.get(key, (0.0, 0.0))
                      for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    vel_ranges = torch.tensor(vel_range_list, device=asset.device)
    vel_samples = math_utils.sample_uniform(
        vel_ranges[:, 0], vel_ranges[:, 1], (n, 6), device=asset.device)
    velocities = root_states[:, 7:13] + vel_samples

    asset.write_root_pose_to_sim(
        torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def smooth_reset_joints_v3(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    position_range: list[float],
    velocity_range: list[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Same as v2's smooth joint reset."""
    asset: Articulation = env.scene[asset_cfg.name]
    state = getattr(llm_curriculum_v3, "_state", None)

    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()

    if state is not None and not state.get("warmup", True):
        d = state["difficulty"].difficulties[env_ids]
        for i in range(len(env_ids)):
            r = _joint_reset_range_schedule(d[i].item())
            joint_pos[i] += torch.empty_like(joint_pos[i]).uniform_(-r, r)
    else:
        noise = math_utils.sample_uniform(
            position_range[0], position_range[1], joint_pos.shape, joint_pos.device)
        joint_pos += noise

    joint_pos = joint_pos.clamp_(
        asset.data.joint_pos_limits[env_ids, :, 0],
        asset.data.joint_pos_limits[env_ids, :, 1])
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


# ──────────────────────────────────────────────────────────────────────────────
# Main curriculum function
# ──────────────────────────────────────────────────────────────────────────────
def llm_curriculum_v3(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
) -> dict[str, float]:
    """LLM Curriculum v3: v2 + adaptive failure-biased spatial sampling."""
    if not hasattr(llm_curriculum_v3, "_state"):
        llm_curriculum_v3._state = {
            "difficulty": PerEnvDifficulty(
                num_envs=env.num_envs, device=env.device,
                step_up=0.04, step_down=0.01,
                promotion_only_until=0.5,
            ),
            "failure": SpatialFailureTracker(
                device=env.device, num_envs=env.num_envs,
                x_bins=4, y_bins=4, z_bins=4, ema_alpha=0.05,
            ),
            "warmup": True,
            "warmup_calls": 0,
            "call_count": 0,
        }

    state = llm_curriculum_v3._state
    tracker = state["difficulty"]
    failure = state["failure"]
    state["call_count"] += 1

    if isinstance(env_ids, torch.Tensor):
        reset_ids = env_ids
    elif isinstance(env_ids, slice):
        reset_ids = torch.arange(env.num_envs, device=env.device)
    else:
        reset_ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)

    # Update outcomes
    if reset_ids.numel() > 0 and not state["warmup"]:
        robot: Articulation = env.scene["robot"]
        obj: RigidObject = env.scene["object"]
        command = env.command_manager.get_command("object_pose")
        des_pos_w, _ = combine_frame_transforms(
            robot.data.root_pos_w[reset_ids], robot.data.root_quat_w[reset_ids],
            command[reset_ids, :3], command[reset_ids, 3:7]
        )
        pos_dist = torch.norm(des_pos_w - obj.data.root_pos_w[reset_ids], dim=1)
        success = pos_dist < 0.15
        tracker.update(reset_ids, success)
        failure.record_outcomes(reset_ids, success)

    if state["warmup"]:
        state["warmup_calls"] += 1
        if state["warmup_calls"] >= 3:
            state["warmup"] = False

    # Apply gravity (scene-wide)
    mean_d = tracker.mean_difficulty
    gz = _gravity_schedule(mean_d)
    env.event_manager.cfg.variable_gravity.params["gravity_distribution_params"] = (
        (0.0, 0.0, gz), (0.0, 0.0, gz)
    )

    # Apply goal range (global)
    gx = _goal_x_schedule(mean_d)
    gy = _goal_y_schedule(mean_d)
    gz_goal = _goal_z_schedule(mean_d)
    cmd_cfg = env.command_manager.cfg.object_pose
    cmd_cfg.ranges.pos_x = gx
    cmd_cfg.ranges.pos_y = gy
    cmd_cfg.ranges.pos_z = gz_goal

    # Failure tracker stats
    weights = failure.get_sampling_weights()
    fr_max = (1.0 - failure.success_ema).max().item()
    fr_min = (1.0 - failure.success_ema).min().item()
    bins_explored = (failure.bin_counts > 0).sum().item()
    entropy = -(weights * (weights + 1e-8).log()).sum().item()
    max_entropy = math.log(failure.total_bins) if failure.total_bins > 0 else 1.0
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

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
        "failure_rate_max": fr_max,
        "failure_rate_min": fr_min,
        "bins_explored": bins_explored,
        "sampling_entropy": norm_entropy,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Config classes
# ──────────────────────────────────────────────────────────────────────────────
@configclass
class LLMv3CurriculumCfg:
    """LLM Curriculum v3 — per-env smooth + adaptive failure sampling."""
    llm_v3 = CurrTerm(func=llm_curriculum_v3)


@configclass
class LLMv3EventCfg(DexsuiteEventCfg):
    """Events with failure-biased per-env difficulty resets."""

    reset_object = EventTerm(
        func=adaptive_smooth_reset_object,
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
        func=smooth_reset_joints_v3,
        mode="reset",
        params={
            "position_range": [-0.1, 0.1],
            "velocity_range": [0.0, 0.0],
        },
    )
