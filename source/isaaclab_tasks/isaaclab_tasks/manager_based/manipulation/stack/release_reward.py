"""
release_reward.py — RELEASE skill (final skill of sequential chain)

Train RELEASE: after stacking cube1 on cube2, open gripper, retract arm to rest.
Resumes from PLACE checkpoint — robot already knows how to pick, transport, place, stack.

Single reward, single objective: let go of the cube and come to a complete rest.

Per-skill curriculum:
  d=0 → cube1 already stacked on cube2, gripper open, arm near rest position.
         Agent barely needs to do anything — just hold still.
  d=1 → cube1 stacked on cube2, gripper CLOSED around it, arm near stack.
         Full release task: open gripper, retract, stop.

Success: gripper open, arm retracted, robot stopped, cube still stacked.
"""

from __future__ import annotations
import json
import math
import time
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import FrameTransformer
from isaaclab.utils import configclass
from isaaclab.envs.mdp import reset_scene_to_default

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ─── Scene Constants ───────────────────────────────────────────
CUBE_HEIGHT = 0.046
CUBE_REST_Z = 0.039
GRIPPER_OPEN = 0.04

# Workspace bounds
WS_X_MIN, WS_X_MAX = 0.4, 0.6
WS_Y_MIN, WS_Y_MAX = -0.10, 0.10

# Default arm pose (retracted rest position) — the TARGET for release
DEFAULT_JOINTS = [
    0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741,  # arm
    0.04, 0.04,  # fingers open
]
DEFAULT_JOINTS_ARM = DEFAULT_JOINTS[:7]  # arm only for pose matching

# Near-stack arm pose (arm extended near cube2 stack position)
NEAR_STACK_JOINTS = [
    0.0, 0.15, 0.0, -2.3, 0.0, 2.5, 0.8,  # arm extended
    0.0, 0.0,  # fingers closed
]

ENV_STEP_DT = 1.0 / 60.0
HOLD_STEPS_SUCCESS = int(0.25 / ENV_STEP_DT)  # 15 steps = 0.25s sustained

# Stack thresholds (same as place_reward)
STACK_XY_THRESH = 0.05
STACK_Z_TOLERANCE = 0.03

# ─── Config ────────────────────────────────────────────────────
CONFIG_PATH = "/tmp/llm_trainer_config.json"
_config_cache = {"data": None, "last_read": 0}

def _load_cfg():
    now = time.time()
    if now - _config_cache["last_read"] < 2.0 and _config_cache["data"] is not None:
        return _config_cache["data"]
    try:
        with open(CONFIG_PATH) as f:
            data = json.load(f)
        _config_cache["data"] = data
        _config_cache["last_read"] = now
        return data
    except Exception:
        return _config_cache["data"] or {}


# ─── Shared State ─────────────────────────────────────────────
_gripper_ids_cache = None

def _get_state(env):
    global _gripper_ids_cache
    ee: FrameTransformer = env.scene["ee_frame"]
    cube1: RigidObject = env.scene["cube_1"]
    cube2: RigidObject = env.scene["cube_2"]
    robot: Articulation = env.scene["robot"]

    ee_pos = ee.data.target_pos_w[:, 0, :]
    rfinger = ee.data.target_pos_w[:, 1, :]
    lfinger = ee.data.target_pos_w[:, 2, :]
    finger_center = (rfinger + lfinger) / 2.0
    c1_pos = cube1.data.root_pos_w
    c2_pos = cube2.data.root_pos_w
    origins = env.scene.env_origins

    if _gripper_ids_cache is None:
        _gripper_ids_cache = [
            robot.data.joint_names.index("panda_finger_joint1"),
            robot.data.joint_names.index("panda_finger_joint2"),
        ]
    f1_idx, f2_idx = _gripper_ids_cache
    finger_pos = robot.data.joint_pos[:, [f1_idx, f2_idx]]
    openness = (finger_pos.mean(dim=1) / GRIPPER_OPEN).clamp(0, 1)

    # Heights
    c1_h = c1_pos[:, 2] - origins[:, 2]
    c2_h = c2_pos[:, 2] - origins[:, 2]
    ee_h = ee_pos[:, 2] - origins[:, 2]

    # Stack checks
    c1_to_c2_xy = torch.norm(c1_pos[:, :2] - c2_pos[:, :2], dim=1)
    c1_above_c2 = c1_h - c2_h

    # EE distance from cube1
    ee_to_c1 = torch.norm(c1_pos - ee_pos, dim=1)
    ee_to_c1_xy = torch.norm(c1_pos[:, :2] - ee_pos[:, :2], dim=1)

    # Joint velocity (arm only, 7 DOF)
    joint_vel = torch.norm(robot.data.joint_vel[:, :7], dim=1)

    # Joint position error from rest pose (arm only, 7 DOF)
    rest_pose = torch.tensor(DEFAULT_JOINTS_ARM, device=robot.data.joint_pos.device, dtype=torch.float32)
    joint_pos_error = torch.norm(robot.data.joint_pos[:, :7] - rest_pose.unsqueeze(0), dim=1)

    return {
        "ee_pos": ee_pos, "ee_h": ee_h,
        "finger_center": finger_center,
        "c1_pos": c1_pos, "c2_pos": c2_pos,
        "openness": openness,
        "c1_h": c1_h, "c2_h": c2_h,
        "c1_to_c2_xy": c1_to_c2_xy,
        "c1_above_c2": c1_above_c2,
        "ee_to_c1": ee_to_c1,
        "ee_to_c1_xy": ee_to_c1_xy,
        "joint_vel": joint_vel,
        "joint_pos_error": joint_pos_error,
        "robot": robot,
        "origins": origins,
    }


# ─── Hold counter for success verification ───────────────────
_rest_hold_counter = None
_episode_success = None  # tracks if each env achieved rest during the episode

def _init_trackers(n, device):
    global _rest_hold_counter, _episode_success
    _rest_hold_counter = torch.zeros(n, dtype=torch.long, device=device)
    _episode_success = torch.zeros(n, dtype=torch.bool, device=device)

def _reset_trackers(env_ids):
    global _rest_hold_counter, _episode_success
    if _rest_hold_counter is not None:
        _rest_hold_counter[env_ids] = 0
    if _episode_success is not None:
        _episode_success[env_ids] = False

def _update_rest_check(s):
    """Check rest conditions every step and update hold counter."""
    global _rest_hold_counter, _episode_success
    gripper_open = s["openness"] > 0.6
    near_rest = s["joint_pos_error"] < 1.0
    robot_stopped = s["joint_vel"] < 0.5
    still_stacked = (s["c1_to_c2_xy"] < STACK_XY_THRESH) & \
                    (torch.abs(s["c1_above_c2"] - CUBE_HEIGHT) < STACK_Z_TOLERANCE)
    resting = gripper_open & near_rest & robot_stopped & still_stacked
    _rest_hold_counter[resting] += 1
    _rest_hold_counter[~resting] = 0
    # Mark envs that achieved sustained rest
    newly_succeeded = _rest_hold_counter >= HOLD_STEPS_SUCCESS
    if _episode_success is not None:
        _episode_success |= newly_succeeded


# ─── REWARD: Release ─────────────────────────────────────────

def release_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Dense release reward: open gripper, retract EE, come to rest.

    Three components with clear gradient:
    1. Open gripper (0.25 weight): linear in openness — immediate gradient
    2. Retract EE (0.25 weight): move EE up and away from cube stack
    3. Stillness (0.50 weight): exponential decay on joint velocity,
       gated on gripper being mostly open (don't reward freezing mid-grip)

    PLUS: stack preservation penalty — negative reward if cube1 gets knocked off
    """
    s = _get_state(env)
    if _rest_hold_counter is None:
        _init_trackers(env.num_envs, env.device)
    _update_rest_check(s)

    # ── 1. Open gripper ──
    open_reward = s["openness"].clamp(0, 1) * 0.15

    # ── 2. Rest pose — move arm to default retracted position ──
    # Dense gradient: exponential decay on joint position error from rest pose
    # At default rest pose: error=0 → reward=1.0. At near-stack pose: error~2 → reward~0.02
    rest_pose_reward = torch.exp(-s["joint_pos_error"] / 0.5) * 0.60

    # ── 3. Stillness bonus — small velocity penalty for smoothness ──
    # Only activates when near rest pose (within ~30% error) to avoid conflicting gradients
    near_rest = (s["joint_pos_error"] < 0.8).float()
    stillness = torch.exp(-(s["joint_vel"] ** 2) / (0.5 ** 2))
    still_reward = near_rest * stillness * 0.25

    # ── 4. Stack preservation penalty ──
    # Cube1 should still be on cube2
    aligned = s["c1_to_c2_xy"] < STACK_XY_THRESH
    correct_h = torch.abs(s["c1_above_c2"] - CUBE_HEIGHT) < STACK_Z_TOLERANCE
    still_stacked = (aligned & correct_h).float()
    # Penalty for knocking cube off
    knocked_penalty = (1.0 - still_stacked) * 2.0

    total = open_reward + rest_pose_reward + still_reward - knocked_penalty
    return total


def action_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    raw = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    return raw.clamp(max=5.0)


# ─── CURRICULUM ───────────────────────────────────────────────

def release_curriculum(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    """Release curriculum: d controls how much work the robot needs to do.

    d=0 → gripper open, arm near rest → just hold still
    d=1 → gripper closed around stacked cube, arm near stack → full release task

    Success is tracked DURING the episode (via _episode_success) and 
    read at reset time for curriculum advancement.
    """
    global _rest_hold_counter, _episode_success

    if not hasattr(release_curriculum, "_n"):
        release_curriculum._n = 0
        release_curriculum._difficulty = torch.zeros(env.num_envs, device=env.device)
        release_curriculum._success_ema = 0.0

    release_curriculum._n += 1
    d = release_curriculum._difficulty

    if _rest_hold_counter is None:
        _init_trackers(env.num_envs, env.device)

    if isinstance(env_ids, torch.Tensor):
        ids = env_ids
    else:
        ids = torch.arange(env.num_envs, device=env.device)

    # Read episode-level success (tracked during reward calls)
    success = _episode_success[ids].float() if _episode_success is not None else torch.zeros(len(ids), device=env.device)

    # Reset trackers for envs that are resetting
    _reset_trackers(ids)

    # Curriculum advancement
    cfg = _load_cfg()
    step_up = cfg.get("per_skill_config", {}).get("release", {}).get("step_up", 0.02)
    step_down = cfg.get("per_skill_config", {}).get("release", {}).get("step_down", 0.005)
    promo_only_until = cfg.get("promotion_only_until", 0.3)

    if ids.numel() > 0 and release_curriculum._n > 2:
        succ_ids = success > 0.5
        fail_ids = ~succ_ids
        d[ids[succ_ids]] = (d[ids[succ_ids]] + step_up).clamp(0.0, 1.0)
        demotable = d[ids] > promo_only_until
        demote_mask = fail_ids & demotable
        d[ids[demote_mask]] = (d[ids[demote_mask]] - step_down).clamp(0.0, 1.0)

    release_curriculum._success_ema = 0.99 * release_curriculum._success_ema + 0.01 * success.mean().item()

    if release_curriculum._n % 50 == 0:
        print(f"[RELEASE] iter={release_curriculum._n} success_rate={success.mean():.3f} "
              f"ema={release_curriculum._success_ema:.3f} d_mean={d.mean():.3f} d_max={d.max():.3f}")

    return success.mean().unsqueeze(0)


# ─── EVENTS ──────────────────────────────────────────────────

def release_reset(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    """Reset with release curriculum.

    CRITICAL: d=0 must NOT start with the task already done (robot at rest).
    The robot always starts with gripper CLOSED near the stacked cube.
    Difficulty controls how far from the stack the arm starts.

    Difficulty axis: arm distance from stack position.
      d=0 → gripper closed on stacked cube, arm right at the stack.
             Easy: just open gripper and retract a small amount.
      d=0.5 → gripper closed, arm at stack, cube2 at varied positions.
      d=1.0 → gripper closed, arm at stack, cubes at full workspace range.
             Full release + retract across varied geometry.

    The robot MUST always perform the release action (open + retract).
    """
    global _rest_hold_counter

    reset_scene_to_default(env, env_ids)

    cube1: RigidObject = env.scene["cube_1"]
    cube2: RigidObject = env.scene["cube_2"]
    robot: Articulation = env.scene["robot"]
    n = len(env_ids)
    device = env.device

    if _rest_hold_counter is not None and _rest_hold_counter.shape[0] > 0:
        _rest_hold_counter[env_ids] = 0
    if _episode_success is not None and _episode_success.shape[0] > 0:
        _episode_success[env_ids] = False

    # Get difficulty
    tracker = getattr(release_curriculum, '_difficulty', None)
    if tracker is not None and tracker.mean() > 0.001:
        d_vals = tracker[env_ids]
    else:
        cfg = _load_cfg()
        play_d = cfg.get("play_difficulty", 0.0)
        d_vals = torch.full((n,), play_d, device=device)

    # ── Cube2: position varies with difficulty ──
    # d=0: cube2 at center of workspace
    # d=1: cube2 at random workspace position
    c2_center_x, c2_center_y = 0.50, 0.0
    c2_x = c2_center_x + d_vals * torch.empty(n, device=device).uniform_(-0.10, 0.10)
    c2_y = c2_center_y + d_vals * torch.empty(n, device=device).uniform_(-0.10, 0.10)
    c2_x = c2_x.clamp(WS_X_MIN, WS_X_MAX)
    c2_y = c2_y.clamp(WS_Y_MIN, WS_Y_MAX)
    c2_z = torch.full((n,), CUBE_REST_Z, device=device)

    c2_state = cube2.data.default_root_state[env_ids].clone()
    c2_state[:, 0] = c2_x
    c2_state[:, 1] = c2_y
    c2_state[:, 2] = c2_z
    c2_state[:, 0:3] += env.scene.env_origins[env_ids]
    c2_state[:, 7:13] = 0
    cube2.write_root_pose_to_sim(c2_state[:, :7], env_ids=env_ids)
    cube2.write_root_velocity_to_sim(c2_state[:, 7:13], env_ids=env_ids)

    # ── Cube1: ALWAYS stacked on cube2 ──
    c1_state = cube1.data.default_root_state[env_ids].clone()
    c1_state[:, 0] = c2_x + torch.empty(n, device=device).uniform_(-0.005, 0.005)
    c1_state[:, 1] = c2_y + torch.empty(n, device=device).uniform_(-0.005, 0.005)
    c1_state[:, 2] = c2_z + CUBE_HEIGHT  # stacked
    c1_state[:, 0:3] += env.scene.env_origins[env_ids]
    c1_state[:, 7:13] = 0
    cube1.write_root_pose_to_sim(c1_state[:, :7], env_ids=env_ids)
    cube1.write_root_velocity_to_sim(c1_state[:, 7:13], env_ids=env_ids)

    # ── Robot arm: ALWAYS start near the stack with gripper CLOSED ──
    # This ensures the agent must ALWAYS perform the release action.
    # d=0: arm at NEAR_STACK pose, gripper closed
    # d=1: same (difficulty is in cube positions, not arm pose)
    near_joints = torch.tensor(NEAR_STACK_JOINTS, device=device, dtype=torch.float32)
    target_joints = near_joints.unsqueeze(0).expand(n, -1).clone()

    # Small noise for robustness
    noise = torch.randn(n, 7, device=device) * 0.02
    target_joints[:, :7] += noise

    # Clamp to joint limits
    joint_limits = robot.data.soft_joint_pos_limits[env_ids]
    target_joints = target_joints.clamp(joint_limits[..., 0], joint_limits[..., 1])

    # Gripper ALWAYS closed — agent must learn to open
    target_joints[:, 7] = 0.0
    target_joints[:, 8] = 0.0

    joint_vel = torch.zeros_like(target_joints)
    robot.set_joint_position_target(target_joints, env_ids=env_ids)
    robot.set_joint_velocity_target(joint_vel, env_ids=env_ids)
    robot.write_joint_state_to_sim(target_joints, joint_vel, env_ids=env_ids)

    # DIAGNOSTIC
    if not hasattr(release_reset, '_diag_count'):
        release_reset._diag_count = 0
    if release_reset._diag_count < 2:
        release_reset._diag_count += 1
        i = 0
        print(f"\nRELEASE RESET DIAG (call {release_reset._diag_count}):")
        print(f"  d_val[0] = {d_vals[i].item():.3f}")
        print(f"  gripper[0] = {target_joints[i, 7].item():.4f} (ALWAYS CLOSED)")
        print(f"  cube1 z = {c1_state[i, 2].item():.4f}")
        print(f"  cube2 pos = ({c2_x[i].item():.3f}, {c2_y[i].item():.3f})")
        print()


# ─── CONFIG CLASSES ───────────────────────────────────────────

@configclass
class ReleaseRewardsCfg:
    release = RewTerm(func=release_reward, weight=1.0)
    action_rate = RewTerm(func=action_penalty, weight=-0.05)


@configclass
class ReleaseCurriculumCfg:
    skill_curriculum = CurrTerm(func=release_curriculum)


@configclass
class ReleaseEventCfg:
    curriculum_reset = EventTerm(func=release_reset, mode="reset")
