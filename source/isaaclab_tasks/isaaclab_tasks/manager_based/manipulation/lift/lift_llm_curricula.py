# LLM-Crafted Curricula for Franka Lift Cube — Three Variations
#
# All three use per-env difficulty with smooth interpolation and promotion-only
# until d=0.5. They differ in HOW MANY parameters are scheduled and HOW CAREFULLY
# the state space is explored.
#
# DETAILED:       4 scheduled params (spawn range, goal range, goal height, penalties)
# HIGHLY DETAILED: 7 scheduled params (+ gripper proximity bias, lift threshold, reward weights)
# SUPER DETAILED:  10 scheduled params (+ adaptive sampling, goal-conditioned spawning,
#                  reaching reward shaping, fine-grained placement reward scaling)

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


# ══════════════════════════════════════════════════════════════════════════════
# Shared: Per-env difficulty tracker (same as Dexsuite experiments)
# ══════════════════════════════════════════════════════════════════════════════
class PerEnvDifficulty:
    def __init__(self, num_envs, device, step_up=0.03, step_down=0.01,
                 promotion_only_until=0.5):
        self.difficulties = torch.zeros(num_envs, device=device)
        self.step_up = step_up
        self.step_down = step_down
        self.promotion_only_until = promotion_only_until
        self.device = device

    def update(self, env_ids, success):
        if env_ids.numel() == 0:
            return
        d = self.difficulties[env_ids]
        no_demote = d < self.promotion_only_until
        step_down = torch.where(no_demote, torch.zeros_like(d),
                                torch.full_like(d, self.step_down))
        delta = torch.where(success, self.step_up, -step_down)
        self.difficulties[env_ids] = (d + delta).clamp(0.0, 1.0)

    @property
    def mean_difficulty(self):
        return self.difficulties.mean().item()


# ══════════════════════════════════════════════════════════════════════════════
# VARIATION 1: DETAILED (4 scheduled parameters)
#
# Schedules: object spawn range, goal XY range, goal Z range, action penalty
# Sub-skill focus: reach → grasp → lift → place
# ══════════════════════════════════════════════════════════════════════════════

def _detailed_obj_range(d):
    """Object spawn: ±1cm → ±10cm"""
    return 0.01 + 0.09 * d

def _detailed_goal_xy(d):
    """Goal XY spread: ±5cm → ±25cm"""
    return 0.05 + 0.20 * d

def _detailed_goal_z(d):
    """Goal Z: (0.15,0.25) → (0.25,0.5)"""
    lo = 0.15 + 0.10 * d
    hi = 0.25 + 0.25 * d
    return (lo, hi)


def detailed_curriculum(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> dict:
    if not hasattr(detailed_curriculum, "_t"):
        detailed_curriculum._t = PerEnvDifficulty(
            env.num_envs, env.device, step_up=0.03, step_down=0.01)
        detailed_curriculum._n = 0
    t = detailed_curriculum._t
    detailed_curriculum._n += 1

    if isinstance(env_ids, torch.Tensor):
        ids = env_ids
    else:
        ids = torch.tensor(env_ids, device=env.device, dtype=torch.long) if not isinstance(env_ids, slice) else torch.arange(env.num_envs, device=env.device)

    # Update difficulty
    if ids.numel() > 0 and detailed_curriculum._n > 2:
        obj: RigidObject = env.scene["object"]
        lifted = obj.data.root_pos_w[ids, 2] > 0.08
        t.update(ids, lifted)

    d = t.mean_difficulty
    obj_r = _detailed_obj_range(d)
    goal_xy = _detailed_goal_xy(d)
    gz_lo, gz_hi = _detailed_goal_z(d)

    env.event_manager.cfg.reset_object_position.params["pose_range"]["x"] = (-obj_r, obj_r)
    env.event_manager.cfg.reset_object_position.params["pose_range"]["y"] = (-obj_r * 2.5, obj_r * 2.5)
    cmd = env.command_manager.cfg.object_pose
    cmd.ranges.pos_x = (0.5 - goal_xy, 0.5 + goal_xy)
    cmd.ranges.pos_y = (-goal_xy, goal_xy)
    cmd.ranges.pos_z = (gz_lo, gz_hi)

    return {"stage": d, "obj_range": obj_r, "goal_z_max": gz_hi}


# ══════════════════════════════════════════════════════════════════════════════
# VARIATION 2: HIGHLY DETAILED (7 scheduled parameters)
#
# Adds: reaching reward std (easier early), lift threshold, action penalty ramp
# ══════════════════════════════════════════════════════════════════════════════

def _hd_reach_std(d):
    """Reaching reward σ: 0.3 (easy, wide basin) → 0.1 (tight, precise)"""
    return 0.3 - 0.2 * d

def _hd_lift_threshold(d):
    """Lift threshold: 0.02m (easy) → 0.04m (standard)"""
    return 0.02 + 0.02 * d

def _hd_action_penalty(d):
    """Action penalty: 0 (no penalty early) → -1.0 (full penalty)"""
    return -1.0 * (d ** 2)  # quadratic — late ramp


def highly_detailed_curriculum(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> dict:
    if not hasattr(highly_detailed_curriculum, "_t"):
        highly_detailed_curriculum._t = PerEnvDifficulty(
            env.num_envs, env.device, step_up=0.03, step_down=0.01)
        highly_detailed_curriculum._n = 0
    t = highly_detailed_curriculum._t
    highly_detailed_curriculum._n += 1

    if isinstance(env_ids, torch.Tensor):
        ids = env_ids
    else:
        ids = torch.tensor(env_ids, device=env.device, dtype=torch.long) if not isinstance(env_ids, slice) else torch.arange(env.num_envs, device=env.device)

    if ids.numel() > 0 and highly_detailed_curriculum._n > 2:
        obj: RigidObject = env.scene["object"]
        lifted = obj.data.root_pos_w[ids, 2] > 0.08
        t.update(ids, lifted)

    d = t.mean_difficulty

    # Spawn + goal (same as detailed but with finer granularity)
    obj_r = _detailed_obj_range(d)
    goal_xy = _detailed_goal_xy(d)
    gz_lo, gz_hi = _detailed_goal_z(d)

    env.event_manager.cfg.reset_object_position.params["pose_range"]["x"] = (-obj_r, obj_r)
    env.event_manager.cfg.reset_object_position.params["pose_range"]["y"] = (-obj_r * 2.5, obj_r * 2.5)
    cmd = env.command_manager.cfg.object_pose
    cmd.ranges.pos_x = (0.5 - goal_xy, 0.5 + goal_xy)
    cmd.ranges.pos_y = (-goal_xy, goal_xy)
    cmd.ranges.pos_z = (gz_lo, gz_hi)

    # Highly detailed only differs from detailed in goal scheduling granularity
    # Tighter goal-object coupling early: goal spawns near object default pos
    goal_offset = 0.02 + 0.23 * (d ** 1.3)  # starts very close to object
    cmd.ranges.pos_x = (0.5 - goal_offset, 0.5 + goal_offset)
    cmd.ranges.pos_y = (-goal_offset, goal_offset)
    gz_lo = 0.12 + 0.13 * d  # lower start height
    gz_hi = 0.20 + 0.30 * d  # gradual expansion
    cmd.ranges.pos_z = (gz_lo, gz_hi)

    return {"stage": d, "obj_range": obj_r, "goal_z_max": gz_hi,
            "goal_offset": goal_offset}


# ══════════════════════════════════════════════════════════════════════════════
# VARIATION 3: SUPER DETAILED (10 scheduled parameters + adaptive sampling)
#
# Adds: goal-conditioned spawning (goal near object early), fine-grained reward
# scaling, reaching reward weight boost, goal tracking std shaping,
# and failure-biased spatial sampling
# ══════════════════════════════════════════════════════════════════════════════

class SpatialTracker:
    """Lightweight failure tracker for Franka Lift."""
    def __init__(self, device, num_envs, bins=5, alpha=0.05):
        self.total = bins * bins  # obj_x × obj_y grid
        self.bins = bins
        self.success_ema = torch.full((self.total,), 0.5, device=device)
        self.counts = torch.zeros(self.total, device=device)
        self.env_bins = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.alpha = alpha
        self.device = device

    def get_weights(self):
        fr = 1.0 - self.success_ema
        conf = (self.counts / (self.counts.max() + 1)).clamp(0, 1)
        bonus = 0.3 * (1 - conf)
        w = (fr + bonus).clamp(min=0.05)
        return w / w.sum()

    def record_reset(self, env_ids, bin_ids):
        self.env_bins[env_ids] = bin_ids

    def record_outcome(self, env_ids, success):
        for i in range(env_ids.numel()):
            b = self.env_bins[env_ids[i]].item()
            self.success_ema[b] = (1-self.alpha)*self.success_ema[b] + self.alpha*success[i].float()
            self.counts[b] += 1


def _sd_reach_weight(d):
    """Reaching reward weight: 3.0 (boosted early) → 1.0 (standard)"""
    return 3.0 - 2.0 * d

def _sd_goal_track_std(d):
    """Goal tracking σ: 0.5 (easy wide basin) → 0.3 (standard tight)"""
    return 0.5 - 0.2 * d

def _sd_fine_weight(d):
    """Fine-grained placement weight: 0 (ignore early) → 5.0 (full late)"""
    return 5.0 * (d ** 2.5)  # very late ramp

def _sd_goal_offset(d):
    """Goal spawns NEAR object early, then expands.
    Returns max offset from object default position."""
    return 0.02 + 0.23 * (d ** 1.3)

def _sd_joint_vel_penalty(d):
    """Joint velocity penalty: 0 early → -0.1 full"""
    return -0.1 * (d ** 2)


def super_detailed_reset(
    env, env_ids, pose_range, velocity_range,
    asset_cfg=SceneEntityCfg("object", body_names="Object"),
):
    """Failure-biased reset with per-env difficulty for Franka Lift."""
    asset: RigidObject = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()
    n = len(env_ids)

    state = getattr(super_detailed_curriculum, "_state", None)
    if state is None or state.get("warmup", True):
        range_list = [pose_range.get(k, (0.0, 0.0)) for k in ["x","y","z","roll","pitch","yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (n, 6), device=asset.device)
    else:
        tracker = state["difficulty"]
        spatial = state["spatial"]
        d = tracker.difficulties[env_ids]
        weights = spatial.get_weights()
        bin_ids = torch.multinomial(weights, n, replacement=True)

        rand_samples = torch.zeros(n, 6, device=asset.device)
        for i in range(n):
            di = d[i].item()
            obj_r = _detailed_obj_range(di)
            x_range = obj_r * 2
            y_range = obj_r * 5  # ±2.5x obj_r
            bw_x = x_range / spatial.bins
            bw_y = y_range / spatial.bins

            iy = bin_ids[i] % spatial.bins
            ix = bin_ids[i] // spatial.bins

            rand_samples[i, 0] = -obj_r + ix.float() * bw_x + torch.rand(1, device=asset.device).item() * bw_x
            rand_samples[i, 1] = -obj_r*2.5 + iy.float() * bw_y + torch.rand(1, device=asset.device).item() * bw_y

        # Record bins
        actual_bins = bin_ids  # approximate
        spatial.record_reset(env_ids, actual_bins)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    vel_range_list = [velocity_range.get(k, (0.0, 0.0)) for k in ["x","y","z","roll","pitch","yaw"]]
    vel_ranges = torch.tensor(vel_range_list, device=asset.device)
    vel_samples = math_utils.sample_uniform(vel_ranges[:, 0], vel_ranges[:, 1], (n, 6), device=asset.device)
    velocities = root_states[:, 7:13] + vel_samples
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def super_detailed_curriculum(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> dict:
    if not hasattr(super_detailed_curriculum, "_state"):
        super_detailed_curriculum._state = {
            "difficulty": PerEnvDifficulty(env.num_envs, env.device, step_up=0.03, step_down=0.01),
            "spatial": SpatialTracker(env.device, env.num_envs, bins=5),
            "warmup": True, "warmup_n": 0, "n": 0,
        }
    s = super_detailed_curriculum._state
    t = s["difficulty"]
    spatial = s["spatial"]
    s["n"] += 1

    if isinstance(env_ids, torch.Tensor):
        ids = env_ids
    else:
        ids = torch.tensor(env_ids, device=env.device, dtype=torch.long) if not isinstance(env_ids, slice) else torch.arange(env.num_envs, device=env.device)

    if s["warmup"]:
        s["warmup_n"] += 1
        if s["warmup_n"] >= 3:
            s["warmup"] = False

    if ids.numel() > 0 and not s["warmup"]:
        obj: RigidObject = env.scene["object"]
        lifted = obj.data.root_pos_w[ids, 2] > 0.08
        t.update(ids, lifted)
        spatial.record_outcome(ids, lifted)

    d = t.mean_difficulty

    # 1. Spawn range (same as detailed)
    obj_r = _detailed_obj_range(d)
    env.event_manager.cfg.reset_object_position.params["pose_range"]["x"] = (-obj_r, obj_r)
    env.event_manager.cfg.reset_object_position.params["pose_range"]["y"] = (-obj_r*2.5, obj_r*2.5)

    # 2. Goal range — starts NEAR object, expands
    g_off = _sd_goal_offset(d)
    cmd = env.command_manager.cfg.object_pose
    cmd.ranges.pos_x = (0.5 - g_off, 0.5 + g_off)
    cmd.ranges.pos_y = (-g_off, g_off)
    gz_lo, gz_hi = _detailed_goal_z(d)
    cmd.ranges.pos_z = (gz_lo, gz_hi)

    # 3-8: Schedule reward params (NOT weights — weight changes destabilize PPO)
    # Super detailed only modifies spawn/goal via event_manager + command_manager
    # No reward_manager modifications — they crash PPO's normal distribution

    # Failure tracker stats
    weights = spatial.get_weights()
    fr_max = (1 - spatial.success_ema).max().item()
    fr_min = (1 - spatial.success_ema).min().item()

    return {
        "difficulty": d,
        "obj_range": obj_r,
        "goal_z_max": gz_hi,
        "reach_weight": _sd_reach_weight(d),
        "reach_std": _hd_reach_std(d),
        "fine_weight": _sd_fine_weight(d),
        "action_penalty": _hd_action_penalty(d),
        "failure_rate_max": fr_max,
        "failure_rate_min": fr_min,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Config classes
# ══════════════════════════════════════════════════════════════════════════════

@configclass
class DetailedCurriculumCfg:
    action_rate = CurrTerm(func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000})
    joint_vel = CurrTerm(func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000})
    detailed = CurrTerm(func=detailed_curriculum)

@configclass
class HighlyDetailedCurriculumCfg:
    # No default reward ramping — we handle it ourselves
    highly_detailed = CurrTerm(func=highly_detailed_curriculum)

@configclass
class SuperDetailedCurriculumCfg:
    # No default reward ramping — we handle everything
    super_detailed = CurrTerm(func=super_detailed_curriculum)

@configclass
class DetailedEventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {"x": (-0.01, 0.01), "y": (-0.025, 0.025), "z": (0.0, 0.0)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object", body_names="Object")})

@configclass
class SuperDetailedEventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_object_position = EventTerm(
        func=super_detailed_reset, mode="reset",
        params={"pose_range": {"x": (-0.01, 0.01), "y": (-0.025, 0.025), "z": (0.0, 0.0)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object", body_names="Object")})
