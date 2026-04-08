# Copyright (c) 2024, Super Curriculum Experiment
# "Super Curriculum" — combines progressive stage-based curriculum with
# adaptive failure-aware sampling. Tracks per-region performance and biases
# environment resets toward states where the policy struggles most.
#
# Architecture:
#   1. PerformanceTracker: bins (obj_x, obj_y, goal_z) into a spatial grid,
#      tracks success/failure counts per bin at every reset.
#   2. AdaptiveSampler: converts failure rates into sampling weights so harder
#      regions get more environments allocated to them.
#   3. Progressive stages: still ramp difficulty from easy→hard, but WITHIN
#      each stage we bias toward the hard spots.

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ──────────────────────────────────────────────────────────────────────────────
# Stage definitions (same as original curriculum)
# ──────────────────────────────────────────────────────────────────────────────
_STAGES = [
    {  # Stage 0: Very easy — cube near center, goal close and low
        "obj_x": (-0.02, 0.02),
        "obj_y": (-0.05, 0.05),
        "goal_x": (0.45, 0.55),
        "goal_y": (-0.1, 0.1),
        "goal_z": (0.15, 0.25),
    },
    {  # Stage 1
        "obj_x": (-0.05, 0.05),
        "obj_y": (-0.1, 0.1),
        "goal_x": (0.42, 0.58),
        "goal_y": (-0.15, 0.15),
        "goal_z": (0.2, 0.3),
    },
    {  # Stage 2
        "obj_x": (-0.07, 0.07),
        "obj_y": (-0.15, 0.15),
        "goal_x": (0.4, 0.6),
        "goal_y": (-0.2, 0.2),
        "goal_z": (0.2, 0.4),
    },
    {  # Stage 3
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
_PROMOTION_LIFT_FRAC = 0.3
_MIN_HOLD_CALLS = 50


# ──────────────────────────────────────────────────────────────────────────────
# Performance Tracker — spatial grid that tracks success/failure per region
# ──────────────────────────────────────────────────────────────────────────────
class PerformanceTracker:
    """Tracks episode outcomes binned by initial (obj_x, obj_y, goal_z) regions.
    
    Uses an exponential moving average so recent performance matters more
    than ancient history (the policy is changing, after all).
    """

    def __init__(self, device: torch.device, num_envs: int,
                 obj_x_bins: int = 5, obj_y_bins: int = 5, goal_z_bins: int = 4,
                 ema_alpha: float = 0.05):
        self.device = device
        self.num_envs = num_envs
        self.obj_x_bins = obj_x_bins
        self.obj_y_bins = obj_y_bins
        self.goal_z_bins = goal_z_bins
        self.total_bins = obj_x_bins * obj_y_bins * goal_z_bins
        self.ema_alpha = ema_alpha

        # EMA of success rate per bin (initialize to 0.5 = uncertain)
        self.success_ema = torch.full((self.total_bins,), 0.5, device=device)
        # Count of samples per bin (for confidence)
        self.bin_counts = torch.zeros(self.total_bins, device=device)
        # Per-env: store the initial state used for current episode
        self.env_init_obj_xy = torch.zeros(num_envs, 2, device=device)
        self.env_init_goal_z = torch.zeros(num_envs, device=device)
        self.env_bins = torch.zeros(num_envs, dtype=torch.long, device=device)

    def _get_bin_idx(self, obj_x: torch.Tensor, obj_y: torch.Tensor, goal_z: torch.Tensor,
                     stage_cfg: dict) -> torch.Tensor:
        """Convert continuous (obj_x, obj_y, goal_z) offsets to flat bin index."""
        # Normalize to [0, 1] within the current stage range
        ox_lo, ox_hi = stage_cfg["obj_x"]
        oy_lo, oy_hi = stage_cfg["obj_y"]
        gz_lo, gz_hi = stage_cfg["goal_z"]

        ox_norm = ((obj_x - ox_lo) / max(ox_hi - ox_lo, 1e-6)).clamp(0, 0.999)
        oy_norm = ((obj_y - oy_lo) / max(oy_hi - oy_lo, 1e-6)).clamp(0, 0.999)
        gz_norm = ((goal_z - gz_lo) / max(gz_hi - gz_lo, 1e-6)).clamp(0, 0.999)

        ix = (ox_norm * self.obj_x_bins).long()
        iy = (oy_norm * self.obj_y_bins).long()
        iz = (gz_norm * self.goal_z_bins).long()

        return ix * (self.obj_y_bins * self.goal_z_bins) + iy * self.goal_z_bins + iz

    def record_init_state(self, env_ids: torch.Tensor, obj_xy_offset: torch.Tensor,
                          goal_z: torch.Tensor, stage_cfg: dict):
        """Record the initial state for environments that just reset."""
        self.env_init_obj_xy[env_ids] = obj_xy_offset
        self.env_init_goal_z[env_ids] = goal_z
        self.env_bins[env_ids] = self._get_bin_idx(
            obj_xy_offset[:, 0], obj_xy_offset[:, 1], goal_z, stage_cfg
        )

    def record_outcomes(self, env_ids: torch.Tensor, lifted: torch.Tensor):
        """Update the EMA success rate for the bins these envs came from.
        
        Args:
            env_ids: environments that are resetting (episode just ended)
            lifted: boolean tensor — did this env successfully lift the object?
        """
        if env_ids.numel() == 0:
            return

        bins = self.env_bins[env_ids]
        success = lifted.float()

        # Update EMA per bin (scatter by bin index)
        for i in range(env_ids.numel()):
            b = bins[i].item()
            self.success_ema[b] = (1 - self.ema_alpha) * self.success_ema[b] + self.ema_alpha * success[i]
            self.bin_counts[b] += 1

    def get_sampling_weights(self) -> torch.Tensor:
        """Return per-bin weights proportional to failure rate.
        
        Higher weight = lower success rate = policy struggles here.
        Bins with no data get moderate weight (exploration bonus).
        """
        failure_rate = 1.0 - self.success_ema
        # Exploration bonus: bins with few samples get a boost
        confidence = (self.bin_counts / (self.bin_counts.max() + 1)).clamp(0, 1)
        exploration_bonus = 0.3 * (1.0 - confidence)
        # Combine: failure_rate + exploration bonus, with a floor
        weights = (failure_rate + exploration_bonus).clamp(min=0.05)
        return weights / weights.sum()

    def reset_for_new_stage(self):
        """Reset tracker when promoting to a new stage (bins change meaning)."""
        self.success_ema.fill_(0.5)
        self.bin_counts.zero_()


# ──────────────────────────────────────────────────────────────────────────────
# Adaptive reset event — replaces uniform sampling with failure-biased sampling
# ──────────────────────────────────────────────────────────────────────────────
def adaptive_reset_object(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Reset object with failure-biased sampling.
    
    Instead of uniform sampling within the pose_range, we:
    1. Check the PerformanceTracker for which spatial bins have highest failure
    2. Sample bin indices proportional to failure weight
    3. Sample positions uniformly within the chosen bin
    
    Falls back to uniform sampling if the tracker isn't initialized yet.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()

    # Check if super curriculum state exists
    state = getattr(super_curriculum, "_state", None)
    tracker = getattr(super_curriculum, "_tracker", None)

    if state is None or tracker is None or state.get("warmup", True):
        # Fallback: uniform sampling (during warmup or before init)
        range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    else:
        stage_cfg = _STAGES[state["stage"]]
        n = len(env_ids)

        # Get sampling weights from tracker
        weights = tracker.get_sampling_weights()

        # Sample bin indices proportional to failure weights
        bin_indices = torch.multinomial(weights, n, replacement=True)

        # Convert flat bin index back to (ix, iy, iz)
        iz = bin_indices % tracker.goal_z_bins
        remainder = bin_indices // tracker.goal_z_bins
        iy = remainder % tracker.obj_y_bins
        ix = remainder // tracker.obj_y_bins

        # Sample uniformly within each chosen bin
        ox_lo, ox_hi = stage_cfg["obj_x"]
        oy_lo, oy_hi = stage_cfg["obj_y"]

        ox_range = ox_hi - ox_lo
        oy_range = oy_hi - oy_lo

        bin_w_x = ox_range / tracker.obj_x_bins
        bin_w_y = oy_range / tracker.obj_y_bins

        # Object x, y offsets: bin_start + random within bin
        obj_x = ox_lo + ix.float() * bin_w_x + torch.rand(n, device=asset.device) * bin_w_x
        obj_y = oy_lo + iy.float() * bin_w_y + torch.rand(n, device=asset.device) * bin_w_y

        # Store goal_z bin info for the tracker (goal is set by command manager,
        # but we record what bin was targeted)
        gz_lo, gz_hi = stage_cfg["goal_z"]
        gz_range = gz_hi - gz_lo
        bin_w_z = gz_range / tracker.goal_z_bins
        goal_z_sampled = gz_lo + iz.float() * bin_w_z + torch.rand(n, device=asset.device) * bin_w_z

        # Build the pose samples
        rand_samples = torch.zeros(n, 6, device=asset.device)
        rand_samples[:, 0] = obj_x
        rand_samples[:, 1] = obj_y
        rand_samples[:, 2] = pose_range.get("z", (0.0, 0.0))[0]  # z offset usually 0

        # Record initial states in tracker
        tracker.record_init_state(
            env_ids,
            torch.stack([obj_x, obj_y], dim=-1),
            goal_z_sampled,
            stage_cfg,
        )

        # Also store goal_z targets so the command manager can use them
        if not hasattr(super_curriculum, "_goal_z_targets"):
            super_curriculum._goal_z_targets = torch.zeros(env.num_envs, device=asset.device)
        super_curriculum._goal_z_targets[env_ids] = goal_z_sampled

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    # Velocities: always uniform (or zero)
    vel_range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    vel_ranges = torch.tensor(vel_range_list, device=asset.device)
    vel_samples = math_utils.sample_uniform(vel_ranges[:, 0], vel_ranges[:, 1], (len(env_ids), 6), device=asset.device)
    velocities = root_states[:, 7:13] + vel_samples

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


# ──────────────────────────────────────────────────────────────────────────────
# Super Curriculum — main curriculum term
# ──────────────────────────────────────────────────────────────────────────────
def super_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
) -> dict[str, float]:
    """Super Curriculum: progressive stages + adaptive failure-aware sampling.
    
    Called at every reset. Before the event manager resets the envs, we:
    1. Record outcomes from the ENDING episodes (the envs about to reset)
    2. Check for stage promotion
    3. Update event params for the new stage (adaptive_reset_object reads our tracker)
    """
    # ── Initialize state on first call ──
    if not hasattr(super_curriculum, "_state"):
        super_curriculum._state = {
            "stage": 0,
            "call_count": 0,
            "warmup": True,
            "warmup_calls": 0,
        }
        super_curriculum._tracker = PerformanceTracker(
            device=env.device,
            num_envs=env.num_envs,
            obj_x_bins=5,
            obj_y_bins=5,
            goal_z_bins=4,
            ema_alpha=0.05,
        )

    state = super_curriculum._state
    tracker = super_curriculum._tracker
    state["call_count"] += 1

    # ── Resolve env_ids ──
    if isinstance(env_ids, torch.Tensor):
        reset_ids = env_ids
    elif isinstance(env_ids, slice):
        reset_ids = torch.arange(env.num_envs, device=env.device)
    else:
        reset_ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)

    # ── Record outcomes from ending episodes ──
    if reset_ids.numel() > 0 and not state["warmup"]:
        obj: RigidObject = env.scene["object"]
        obj_heights = obj.data.root_pos_w[reset_ids, 2]
        lifted = obj_heights > 0.08
        tracker.record_outcomes(reset_ids, lifted)

    # ── Warmup: first few calls use uniform sampling to seed the tracker ──
    if state["warmup"]:
        state["warmup_calls"] += 1
        if state["warmup_calls"] >= 3:  # After 3 resets, start adaptive
            state["warmup"] = False

    # ── Check for stage promotion ──
    if reset_ids.numel() > 10:
        obj: RigidObject = env.scene["object"]
        obj_heights = obj.data.root_pos_w[reset_ids, 2]
        lift_fraction = (obj_heights > 0.08).float().mean().item()

        if (
            lift_fraction >= _PROMOTION_LIFT_FRAC
            and state["call_count"] >= _MIN_HOLD_CALLS
            and state["stage"] < _NUM_STAGES - 1
        ):
            state["stage"] += 1
            state["call_count"] = 0
            tracker.reset_for_new_stage()

    # ── Update event params for current stage ──
    stage_cfg = _STAGES[state["stage"]]
    env.event_manager.cfg.reset_object_position.params["pose_range"]["x"] = stage_cfg["obj_x"]
    env.event_manager.cfg.reset_object_position.params["pose_range"]["y"] = stage_cfg["obj_y"]

    # Update goal/command range
    cmd_cfg = env.command_manager.cfg.object_pose
    cmd_cfg.ranges.pos_x = stage_cfg["goal_x"]
    cmd_cfg.ranges.pos_y = stage_cfg["goal_y"]
    cmd_cfg.ranges.pos_z = stage_cfg["goal_z"]

    # ── Compute stats for logging ──
    weights = tracker.get_sampling_weights()
    failure_rate_max = (1.0 - tracker.success_ema).max().item()
    failure_rate_min = (1.0 - tracker.success_ema).min().item()
    bins_explored = (tracker.bin_counts > 0).sum().item()
    weight_entropy = -(weights * (weights + 1e-8).log()).sum().item()
    # Max entropy = log(total_bins) for uniform
    max_entropy = math.log(tracker.total_bins)
    # Normalized: 0 = all weight on one bin, 1 = uniform
    norm_entropy = weight_entropy / max_entropy if max_entropy > 0 else 0.0

    return {
        "stage": state["stage"],
        "obj_x_range": stage_cfg["obj_x"][1],
        "goal_z_max": stage_cfg["goal_z"][1],
        "failure_rate_max": failure_rate_max,
        "failure_rate_min": failure_rate_min,
        "bins_explored": bins_explored,
        "sampling_entropy": norm_entropy,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Config classes
# ──────────────────────────────────────────────────────────────────────────────
@configclass
class SuperCurriculumCfg:
    """Super Curriculum config — progressive stages + adaptive failure sampling."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000},
    )
    super_progressive = CurrTerm(func=super_curriculum)


@configclass
class SuperCurriculumEventCfg:
    """Events for super curriculum — uses adaptive reset instead of uniform."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=adaptive_reset_object,
        mode="reset",
        params={
            "pose_range": {"x": (-0.02, 0.02), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )
