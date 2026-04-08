# Dexsuite Kuka Allegro - Super Curriculum
# Combines the existing ADR curriculum with adaptive failure-aware reset sampling.
# Tracks which spatial regions the policy fails in and over-samples those at reset.

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
from isaaclab.utils.math import combine_frame_transforms

from isaaclab_tasks.manager_based.manipulation.dexsuite import mdp
from isaaclab_tasks.manager_based.manipulation.dexsuite.adr_curriculum import CurriculumCfg as ADRCurriculumCfg
from isaaclab_tasks.manager_based.manipulation.dexsuite.dexsuite_env_cfg import EventCfg as DexsuiteEventCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ──────────────────────────────────────────────────────────────────────────────
# Performance Tracker for Dexsuite
# ──────────────────────────────────────────────────────────────────────────────
class DexsuitePerformanceTracker:
    """Tracks success per spatial bin of (obj_x_offset, obj_y_offset, obj_z_offset).
    
    Object default pos is (-0.55, 0.1, 0.35), reset range is:
        x: [-0.2, 0.2], y: [-0.2, 0.2], z: [0.0, 0.4]
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

        self.success_ema = torch.full((self.total_bins,), 0.5, device=device)
        self.bin_counts = torch.zeros(self.total_bins, device=device)
        self.env_bins = torch.zeros(num_envs, dtype=torch.long, device=device)

        # Object spawn ranges (offsets from default)
        self.obj_x_range = (-0.2, 0.2)
        self.obj_y_range = (-0.2, 0.2)
        self.obj_z_range = (0.0, 0.4)

    def _get_bin_idx(self, ox: torch.Tensor, oy: torch.Tensor, oz: torch.Tensor) -> torch.Tensor:
        ox_norm = ((ox - self.obj_x_range[0]) / (self.obj_x_range[1] - self.obj_x_range[0])).clamp(0, 0.999)
        oy_norm = ((oy - self.obj_y_range[0]) / (self.obj_y_range[1] - self.obj_y_range[0])).clamp(0, 0.999)
        oz_norm = ((oz - self.obj_z_range[0]) / (self.obj_z_range[1] - self.obj_z_range[0])).clamp(0, 0.999)
        ix = (ox_norm * self.x_bins).long()
        iy = (oy_norm * self.y_bins).long()
        iz = (oz_norm * self.z_bins).long()
        return ix * (self.y_bins * self.z_bins) + iy * self.z_bins + iz

    def record_init_state(self, env_ids: torch.Tensor, obj_offsets: torch.Tensor):
        """Record which bin each env's object started in. obj_offsets: (N, 3) xyz offsets."""
        self.env_bins[env_ids] = self._get_bin_idx(
            obj_offsets[:, 0], obj_offsets[:, 1], obj_offsets[:, 2]
        )

    def record_outcomes(self, env_ids: torch.Tensor, success: torch.Tensor):
        if env_ids.numel() == 0:
            return
        bins = self.env_bins[env_ids]
        s = success.float()
        for i in range(env_ids.numel()):
            b = bins[i].item()
            self.success_ema[b] = (1 - self.ema_alpha) * self.success_ema[b] + self.ema_alpha * s[i]
            self.bin_counts[b] += 1

    def get_sampling_weights(self) -> torch.Tensor:
        failure_rate = 1.0 - self.success_ema
        confidence = (self.bin_counts / (self.bin_counts.max() + 1)).clamp(0, 1)
        exploration_bonus = 0.3 * (1.0 - confidence)
        weights = (failure_rate + exploration_bonus).clamp(min=0.05)
        return weights / weights.sum()


# ──────────────────────────────────────────────────────────────────────────────
# Adaptive reset for Dexsuite objects
# ──────────────────────────────────────────────────────────────────────────────
def adaptive_reset_dexsuite_object(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Failure-biased object reset for Dexsuite environments."""
    asset: RigidObject = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()

    tracker = getattr(dexsuite_super_curriculum, "_tracker", None)
    state = getattr(dexsuite_super_curriculum, "_state", None)

    if tracker is None or state is None or state.get("warmup", True):
        # Fallback: uniform sampling
        range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    else:
        n = len(env_ids)
        weights = tracker.get_sampling_weights()
        bin_indices = torch.multinomial(weights, n, replacement=True)

        # Convert flat bin to (ix, iy, iz)
        iz = bin_indices % tracker.z_bins
        remainder = bin_indices // tracker.z_bins
        iy = remainder % tracker.y_bins
        ix = remainder // tracker.y_bins

        # Sample within bins
        x_lo, x_hi = pose_range.get("x", (-0.2, 0.2))
        y_lo, y_hi = pose_range.get("y", (-0.2, 0.2))
        z_lo, z_hi = pose_range.get("z", (0.0, 0.4))

        bw_x = (x_hi - x_lo) / tracker.x_bins
        bw_y = (y_hi - y_lo) / tracker.y_bins
        bw_z = (z_hi - z_lo) / tracker.z_bins

        obj_x = x_lo + ix.float() * bw_x + torch.rand(n, device=asset.device) * bw_x
        obj_y = y_lo + iy.float() * bw_y + torch.rand(n, device=asset.device) * bw_y
        obj_z = z_lo + iz.float() * bw_z + torch.rand(n, device=asset.device) * bw_z

        # Rotations: still uniform
        roll_range = pose_range.get("roll", (0.0, 0.0))
        pitch_range = pose_range.get("pitch", (0.0, 0.0))
        yaw_range = pose_range.get("yaw", (0.0, 0.0))

        rand_samples = torch.zeros(n, 6, device=asset.device)
        rand_samples[:, 0] = obj_x
        rand_samples[:, 1] = obj_y
        rand_samples[:, 2] = obj_z
        rand_samples[:, 3] = torch.empty(n, device=asset.device).uniform_(*roll_range)
        rand_samples[:, 4] = torch.empty(n, device=asset.device).uniform_(*pitch_range)
        rand_samples[:, 5] = torch.empty(n, device=asset.device).uniform_(*yaw_range)

        # Record for tracker
        tracker.record_init_state(env_ids, torch.stack([obj_x, obj_y, obj_z], dim=-1))

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    vel_range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    vel_ranges = torch.tensor(vel_range_list, device=asset.device)
    vel_samples = math_utils.sample_uniform(vel_ranges[:, 0], vel_ranges[:, 1], (len(env_ids), 6), device=asset.device)
    velocities = root_states[:, 7:13] + vel_samples

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


# ──────────────────────────────────────────────────────────────────────────────
# Super Curriculum term for Dexsuite
# ──────────────────────────────────────────────────────────────────────────────
def dexsuite_super_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
) -> dict[str, float]:
    """Track performance and update adaptive sampling for Dexsuite lift task."""
    if not hasattr(dexsuite_super_curriculum, "_state"):
        dexsuite_super_curriculum._state = {
            "warmup": True,
            "warmup_calls": 0,
        }
        dexsuite_super_curriculum._tracker = DexsuitePerformanceTracker(
            device=env.device,
            num_envs=env.num_envs,
            x_bins=4, y_bins=4, z_bins=4,
            ema_alpha=0.05,
        )

    state = dexsuite_super_curriculum._state
    tracker = dexsuite_super_curriculum._tracker

    if isinstance(env_ids, torch.Tensor):
        reset_ids = env_ids
    elif isinstance(env_ids, slice):
        reset_ids = torch.arange(env.num_envs, device=env.device)
    else:
        reset_ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)

    # Record outcomes — check if object is near goal
    if reset_ids.numel() > 0 and not state["warmup"]:
        from isaaclab.assets import Articulation
        robot: Articulation = env.scene["robot"]
        obj: RigidObject = env.scene["object"]
        command = env.command_manager.get_command("object_pose")
        des_pos_w, _ = combine_frame_transforms(
            robot.data.root_pos_w[reset_ids], robot.data.root_quat_w[reset_ids],
            command[reset_ids, :3], command[reset_ids, 3:7]
        )
        pos_dist = torch.norm(des_pos_w - obj.data.root_pos_w[reset_ids], dim=1)
        success = pos_dist < 0.15  # within 15cm of goal = success
        tracker.record_outcomes(reset_ids, success)

    if state["warmup"]:
        state["warmup_calls"] += 1
        if state["warmup_calls"] >= 3:
            state["warmup"] = False

    weights = tracker.get_sampling_weights()
    failure_rate_max = (1.0 - tracker.success_ema).max().item()
    failure_rate_min = (1.0 - tracker.success_ema).min().item()
    bins_explored = (tracker.bin_counts > 0).sum().item()
    weight_entropy = -(weights * (weights + 1e-8).log()).sum().item()
    max_entropy = math.log(tracker.total_bins) if tracker.total_bins > 0 else 1.0
    norm_entropy = weight_entropy / max_entropy if max_entropy > 0 else 0.0

    return {
        "failure_rate_max": failure_rate_max,
        "failure_rate_min": failure_rate_min,
        "bins_explored": bins_explored,
        "sampling_entropy": norm_entropy,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Config classes
# ──────────────────────────────────────────────────────────────────────────────
@configclass
class DexsuiteSuperCurriculumCfg(ADRCurriculumCfg):
    """ADR + adaptive failure sampling."""
    adaptive_sampling = CurrTerm(func=dexsuite_super_curriculum)


@configclass
class DexsuiteSuperEventCfg(DexsuiteEventCfg):
    """Override object reset with adaptive sampling."""

    reset_object = EventTerm(
        func=adaptive_reset_dexsuite_object,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.2, 0.2],
                "y": [-0.2, 0.2],
                "z": [0.0, 0.4],
                "roll": [-3.14, 3.14],
                "pitch": [-3.14, 3.14],
                "yaw": [-3.14, 3.14],
            },
            "velocity_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
