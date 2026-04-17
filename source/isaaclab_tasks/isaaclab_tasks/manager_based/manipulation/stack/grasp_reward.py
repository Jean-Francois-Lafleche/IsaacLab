"""
grasp_reward.py — PLACE skill (Skill 3 of sequential chain) — FIXED CURRICULUM

Train PLACE: transport cube1 to above cube2 and release so it stacks.
Resumes from GRASP checkpoint — robot already knows how to reach + grasp.

Single-skill paradigm: ONE primary reward (place) + small maintenance rewards
for approach (w=0.2) and grasp (w=0.3) so the robot doesn't forget.

FIXED Per-skill curriculum (both cubes ALWAYS on table):
  d=0 → cube1 and cube2 on the table, 5cm apart (MIN_CUBE_SEPARATION).
         Robot must pick cube1, short transport, place on cube2.
         Easy because transport distance is minimal.
  d=1 → cube1 and cube2 at full workspace distance (~28cm apart).
         Full pick-transport-place required.

CRITICAL: At NO difficulty level is cube1 pre-stacked. The robot ALWAYS
has to pick up cube1 and place it on cube2. Difficulty only controls
how far apart the cubes start on the table.

Includes hold-penalty: ramping negative reward if robot hovers above target
with gripper closed for >30 steps (prevents holding exploit).
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
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils import configclass
from isaaclab.envs.mdp import reset_scene_to_default

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ─── Scene Constants ───────────────────────────────────────────
CUBE_HEIGHT = 0.046
CUBE_REST_Z = 0.039  # ACTUAL root z relative to env_origins
GRIPPER_OPEN = 0.04

# Workspace bounds
WS_X_MIN, WS_X_MAX = 0.4, 0.6
WS_Y_MIN, WS_Y_MAX = -0.10, 0.10

# Cube separation bounds for curriculum
MIN_CUBE_SEPARATION = 0.05   # d=0: cubes 5cm apart (close but NOT stacked)
MAX_CUBE_SEPARATION = 0.25   # d=1: cubes ~25cm apart (near full workspace diagonal)

# Default arm pose (standard start position — arm retracted above table)
DEFAULT_JOINTS = [
    0.0,        # panda_joint1
    -0.569,     # panda_joint2
    0.0,        # panda_joint3
    -2.810,     # panda_joint4
    0.0,        # panda_joint5
    3.037,      # panda_joint6
    0.741,      # panda_joint7
    0.04,       # panda_finger_joint1
    0.04,       # panda_finger_joint2
]

# Pre-positioned arm pose (fingers near cube, from grasp skill)
NEAR_CUBE_JOINTS = [
    0.0,       # panda_joint1
    0.15,      # panda_joint2
    0.0,       # panda_joint3
    -2.3,      # panda_joint4
    0.0,       # panda_joint5
    2.5,       # panda_joint6
    0.8,       # panda_joint7
    0.04,      # panda_finger_joint1
    0.04,      # panda_finger_joint2
]

# Env timing: sim.dt=1/120, decimation=2 → env_step_dt=1/60
ENV_STEP_DT = 1.0 / 60.0
HOLD_STEPS_SUCCESS = int(0.5 / ENV_STEP_DT)  # 30 steps = 0.5s for stack success verification

# Stack target height: cube1 resting on top of cube2
STACK_TARGET_Z = CUBE_REST_Z + CUBE_HEIGHT  # 0.039 + 0.046 = 0.085

# Stack success thresholds (VALIDATED against video)
STACK_XY_THRESH = 0.05       # 5cm horizontal alignment
STACK_Z_TOLERANCE = 0.03     # ±3cm height tolerance
GRIPPER_RELEASE_THRESH = 0.6 # 60% open = released
CUBE_SPEED_THRESH = 0.3      # allows settling vibration

# Hold penalty parameters
HOLD_PENALTY_ONSET = 30      # start penalizing after 30 steps of hovering

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

    ee_pos = ee.data.target_pos_w[:, 0, :]       # wrist
    rfinger = ee.data.target_pos_w[:, 1, :]       # right fingertip
    lfinger = ee.data.target_pos_w[:, 2, :]       # left fingertip
    c1_pos = cube1.data.root_pos_w
    c1_vel = cube1.data.root_lin_vel_w
    c2_pos = cube2.data.root_pos_w
    c2_vel = cube2.data.root_lin_vel_w
    origins = env.scene.env_origins

    # Finger center for ALL distance computations
    finger_center = (rfinger + lfinger) / 2.0
    c1_to_fingers = torch.norm(c1_pos - finger_center, dim=1)

    # Finger joint positions for openness
    if _gripper_ids_cache is None:
        _gripper_ids_cache = [
            robot.data.joint_names.index("panda_finger_joint1"),
            robot.data.joint_names.index("panda_finger_joint2"),
        ]
    f1_idx, f2_idx = _gripper_ids_cache
    finger_pos = robot.data.joint_pos[:, [f1_idx, f2_idx]]  # (N, 2)
    openness = finger_pos.mean(dim=1) / GRIPPER_OPEN  # 0=closed, 1=open

    # Grasp verification: cube Y between left and right fingertip Y (spatial bracket)
    cube_between_y = (
        ((rfinger[:, 1] < c1_pos[:, 1]) & (c1_pos[:, 1] < lfinger[:, 1])) |
        ((lfinger[:, 1] < c1_pos[:, 1]) & (c1_pos[:, 1] < rfinger[:, 1]))
    )
    z_ok = torch.abs(c1_pos[:, 2] - finger_center[:, 2]) < 0.05
    closing = openness < 0.7
    grasped = cube_between_y & z_ok & closing

    # Heights relative to env_origins
    c1_h = c1_pos[:, 2] - origins[:, 2]
    c2_h = c2_pos[:, 2] - origins[:, 2]

    # Stack target: directly above cube2
    stack_target = c2_pos.clone()
    stack_target[:, 2] = stack_target[:, 2] + CUBE_HEIGHT

    # 3D distance from cube1 to stack target
    c1_to_target_3d = torch.norm(c1_pos - stack_target, dim=1)

    # XY distance from cube1 to cube2
    c1_to_c2_xy = torch.norm(c1_pos[:, :2] - c2_pos[:, :2], dim=1)

    # Height difference: how close cube1 is to target height
    c1_c2_z_diff = c1_h - c2_h  # should be ~CUBE_HEIGHT when stacked

    return {
        "ee_pos": ee_pos,
        "rfinger": rfinger,
        "lfinger": lfinger,
        "finger_center": finger_center,
        "c1_pos": c1_pos,
        "c1_vel": c1_vel,
        "c2_pos": c2_pos,
        "c2_vel": c2_vel,
        "c1_to_fingers": c1_to_fingers,
        "finger_pos": finger_pos,
        "openness": openness,
        "grasped": grasped,
        "origins": origins,
        "robot": robot,
        "c1_h": c1_h,
        "c2_h": c2_h,
        "stack_target": stack_target,
        "c1_to_target_3d": c1_to_target_3d,
        "c1_to_c2_xy": c1_to_c2_xy,
        "c1_c2_z_diff": c1_c2_z_diff,
    }


# ─── Hold-penalty counter (per-env, persistent across steps) ──
_hold_above_counter = None


# ─── REWARD: Place (primary active reward) ─────────────────────

def place_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Dense place reward for stacking cube1 on top of cube2.

    V3 — key changes:
    - Transport reward stays strong all the way to target (no diminish)
    - Release is gated on XY alignment — only reward opening when properly positioned
    - Hold-penalty aggressive: forces release within 20 steps of arrival
    - Stack bonus very large (10.0/step) to dominate all other rewards
    """
    global _hold_above_counter

    s = _get_state(env)
    device = s["c1_pos"].device
    N = s["c1_pos"].shape[0]

    if _hold_above_counter is None:
        _hold_above_counter = torch.zeros(N, device=device)
    elif _hold_above_counter.shape[0] != N:
        _hold_above_counter = torch.zeros(N, device=device)

    dist = s["c1_to_fingers"]
    openness = s["openness"]
    grasped = s["grasped"]

    # ── 1. MAINTENANCE: approach (finger-center → cube1) ──
    # Fade out maintenance rewards when cube1 is near the stack target
    # so they don't compete with transport/release/stack signals
    transport_dist = s["c1_to_target_3d"]
    near_target = transport_dist < 0.08
    maint_fade = (1.0 - torch.exp(-transport_dist / 0.10)).clamp(0.0, 1.0)  # 1.0 far, ~0 at target
    approach = (1.0 - torch.tanh(dist / 0.08)) * 0.2 * maint_fade

    # ── 2. MAINTENANCE: grasp verification bonus ──
    grasp_maint = grasped.float() * 0.3 * maint_fade

    # ── 3. PRIMARY: transport (cube1 → stack target) ──
    # Dense shaping: wider sigma (0.12) for gradient at realistic distances (5-25cm).
    # Plus a secondary coarse term for very-long-range signal.
    transport_fine = torch.exp(-transport_dist / 0.12)
    transport_coarse = (1.0 - torch.tanh(transport_dist / 0.20)) * 0.5  # broad signal
    transport = transport_fine + transport_coarse
    # Gate: must be grasped OR already near target (during release)
    transport_gated = (grasped | near_target).float() * transport * 3.0

    # ── 4. PRIMARY: release incentive ──
    # GATED on XY alignment: only reward opening when cube1 is above cube2
    xy_dist = s["c1_to_c2_xy"]
    xy_aligned_soft = torch.exp(-xy_dist / 0.03)  # peaks when XY aligned
    height_near = torch.exp(-((s["c1_h"] - STACK_TARGET_Z) ** 2) / (0.02 ** 2))  # peaks at stack height
    position_quality = xy_aligned_soft * height_near  # 0-1, high when well positioned
    release = position_quality * (openness ** 2) * 4.0  # only reward opening when positioned well

    # ── 5. PRIMARY: stack success bonus ──
    c1_speed = torch.norm(s["c1_vel"], dim=1)
    xy_aligned = xy_dist < STACK_XY_THRESH
    height_ok = torch.abs(s["c1_c2_z_diff"] - CUBE_HEIGHT) < STACK_Z_TOLERANCE
    above = s["c1_c2_z_diff"] > 0
    released = openness > GRIPPER_RELEASE_THRESH
    settled = c1_speed < CUBE_SPEED_THRESH
    stacked = xy_aligned & height_ok & above & released & settled
    stack_bonus = stacked.float() * 10.0

    # ── 6. PENALTY: hold-penalty (aggressive) ──
    above_target = (xy_dist < 0.06) & (s["c1_h"] > STACK_TARGET_Z - 0.02)
    still_gripping = openness < 0.5
    holding_above = above_target & still_gripping

    _hold_above_counter[holding_above] += 1
    _hold_above_counter[~holding_above] = 0

    # Onset at 20 steps, full penalty by 40 steps
    penalty_ramp = ((_hold_above_counter - 20).clamp(min=0) / 20.0).clamp(max=1.0)
    hold_penalty = penalty_ramp * 15.0

    # ── 7. PENALTY: bump penalty on cube2 ──
    c2_speed = torch.norm(s["c2_vel"], dim=1)
    bump_c2 = c2_speed * 1.0

    # ── 8. Rest reward: stop jiggling after successful stack ──
    joint_vel_mag = torch.norm(s["robot"].data.joint_vel[:, :7], dim=1)
    vel_reward = torch.exp(-(joint_vel_mag ** 2) / (0.5 ** 2))
    rest = stacked.float() * vel_reward * 2.0

    total = (
        approach
        + grasp_maint
        + transport_gated
        + release
        + stack_bonus
        + rest
        - hold_penalty
        - bump_c2
    )

    return total


def action_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Clamped action penalty for smooth movements."""
    raw = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    return raw.clamp(max=5.0)


# ─── CURRICULUM ───────────────────────────────────────────────

_stack_hold_counter = None

def place_curriculum(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    """Place curriculum with stack success advancement gate.

    FIXED: Difficulty controls cube1-to-cube2 DISTANCE ON THE TABLE.
    Both cubes are ALWAYS on the table. Robot ALWAYS has to pick, transport, place.

    d=0   → cubes 5cm apart on the table (short transport)
    d=1.0 → cubes ~25cm apart on the table (full workspace transport)

    Success = cube stacked (aligned, correct height, released, settled) for 0.5s.
    """
    global _stack_hold_counter

    if not hasattr(place_curriculum, "_n"):
        place_curriculum._n = 0
        place_curriculum._difficulty = torch.zeros(env.num_envs, device=env.device)
        place_curriculum._success_ema = 0.0

    place_curriculum._n += 1
    d = place_curriculum._difficulty

    if _stack_hold_counter is None:
        _stack_hold_counter = torch.zeros(env.num_envs, device=env.device)

    if isinstance(env_ids, torch.Tensor):
        ids = env_ids
    else:
        ids = torch.arange(env.num_envs, device=env.device)

    # Stack success check
    with torch.no_grad():
        s = _get_state(env)
        c1_speed = torch.norm(s["c1_vel"], dim=1)
        xy_aligned = s["c1_to_c2_xy"] < STACK_XY_THRESH
        height_ok = torch.abs(s["c1_c2_z_diff"] - CUBE_HEIGHT) < STACK_Z_TOLERANCE
        above = s["c1_c2_z_diff"] > 0
        released = s["openness"] > GRIPPER_RELEASE_THRESH
        settled = c1_speed < CUBE_SPEED_THRESH
        stacked = xy_aligned & height_ok & above & released & settled

        _stack_hold_counter[stacked] += 1
        _stack_hold_counter[~stacked] = 0

        success = (_stack_hold_counter >= HOLD_STEPS_SUCCESS).float()

    # Reset counter for envs that just reset
    if ids.numel() > 0:
        _stack_hold_counter[ids] = 0

    # Load config for tunable params
    cfg = _load_cfg()
    step_up = cfg.get("per_skill_config", {}).get("place", {}).get("step_up", 0.02)
    step_down = cfg.get("per_skill_config", {}).get("place", {}).get("step_down", 0.005)
    promo_only_until = cfg.get("promotion_only_until", 0.3)

    if ids.numel() > 0 and place_curriculum._n > 2:
        succ_ids = success[ids] > 0.5
        fail_ids = ~succ_ids
        d[ids[succ_ids]] = (d[ids[succ_ids]] + step_up).clamp(0.0, 1.0)
        demotable = d[ids] > promo_only_until
        demote_mask = fail_ids & demotable
        d[ids[demote_mask]] = (d[ids[demote_mask]] - step_down).clamp(0.0, 1.0)

    # Update success EMA
    place_curriculum._success_ema = 0.99 * place_curriculum._success_ema + 0.01 * success.mean().item()

    # Logging every 50 curriculum calls
    if place_curriculum._n % 50 == 0:
        print(f"[PLACE] iter={place_curriculum._n} success_rate={success.mean():.3f} "
              f"ema={place_curriculum._success_ema:.3f} d_mean={d.mean():.3f} d_max={d.max():.3f}")

    return success.mean().unsqueeze(0)


# ─── EVENTS ──────────────────────────────────────────────────

def place_reset(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    """Reset with FIXED place curriculum.

    CRITICAL FIX: Both cubes ALWAYS start on the table. NO pre-stacking.

    Difficulty axis: cube1-to-cube2 horizontal distance.
      d=0:   Cube1 and cube2 on the table, 5cm apart (MIN_CUBE_SEPARATION).
             Short transport — robot picks cube1, moves it a tiny bit, places on cube2.
      d=0.5: Cubes ~15cm apart on table. Medium transport.
      d=1.0: Cubes ~25cm apart on table. Full workspace transport.

    Distribution-based: d controls the UPPER BOUND of the separation range.
    Actual separation sampled from Uniform(MIN_CUBE_SEPARATION, MIN + d*(MAX-MIN)).

    Robot arm starts at default retracted pose. Gripper open.
    The robot MUST reach, grasp, lift, transport, and place at ALL difficulty levels.
    """
    global _hold_above_counter

    reset_scene_to_default(env, env_ids)

    cube1: RigidObject = env.scene["cube_1"]
    cube2: RigidObject = env.scene["cube_2"]
    robot: Articulation = env.scene["robot"]
    n = len(env_ids)
    device = env.device

    # Reset hold-above counter for reset envs
    if _hold_above_counter is not None and _hold_above_counter.shape[0] > 0:
        _hold_above_counter[env_ids] = 0

    # --- Get difficulty ---
    tracker = getattr(place_curriculum, '_difficulty', None)
    if tracker is not None and tracker.mean() > 0.001:
        d_vals = tracker[env_ids]
    else:
        cfg = _load_cfg()
        play_d = cfg.get("play_difficulty", 0.0)  # default 0 for training, set to 1.0 in config for play
        d_vals = torch.full((n,), play_d, device=device)

    # Distribution-based: upper bound of separation distance
    # d=0 → upper = MIN (always 5cm)
    # d=1 → upper = MAX (25cm)
    upper_sep = MIN_CUBE_SEPARATION + d_vals * (MAX_CUBE_SEPARATION - MIN_CUBE_SEPARATION)
    # Sample actual separation from Uniform(MIN, upper)
    actual_sep = MIN_CUBE_SEPARATION + torch.rand(n, device=device) * (upper_sep - MIN_CUBE_SEPARATION)

    # --- Cube2: random position within workspace ---
    c2_x = torch.empty(n, device=device).uniform_(WS_X_MIN, WS_X_MAX)
    c2_y = torch.empty(n, device=device).uniform_(WS_Y_MIN, WS_Y_MAX)
    c2_z = torch.full((n,), CUBE_REST_Z, device=device)

    c2_state = cube2.data.default_root_state[env_ids].clone()
    c2_state[:, 0] = c2_x
    c2_state[:, 1] = c2_y
    c2_state[:, 2] = c2_z
    c2_state[:, 0:3] += env.scene.env_origins[env_ids]
    c2_state[:, 7:13] = 0
    cube2.write_root_pose_to_sim(c2_state[:, :7], env_ids=env_ids)
    cube2.write_root_velocity_to_sim(c2_state[:, 7:13], env_ids=env_ids)

    # --- Cube1 position ---
    # d=0: cube1 at its DEFAULT position (where NEAR_CUBE_JOINTS fingers are)
    # d=1: cube1 on table at actual_sep from cube2
    angle = torch.empty(n, device=device).uniform_(0, 2 * math.pi)
    
    # Table position at distance from cube2 (d=1)
    c1_table_x = c2_x + actual_sep * torch.cos(angle)
    c1_table_y = c2_y + actual_sep * torch.sin(angle)
    c1_table_z = torch.full((n,), CUBE_REST_Z, device=device)
    
    # Default cube1 position (d=0): at EE finger position when arm at NEAR_CUBE_JOINTS
    # From diagnostic: finger_center at NEAR_CUBE_JOINTS ≈ (0.51, 0.0, ~0.04)
    c1_default_x = torch.full((n,), 0.51, device=device)
    c1_default_y = torch.full((n,), 0.0, device=device)
    c1_default_z = torch.full((n,), CUBE_REST_Z, device=device)  # on table under fingers
    
    # Interpolate
    c1_x = c1_default_x + d_vals * (c1_table_x - c1_default_x)
    c1_y = c1_default_y + d_vals * (c1_table_y - c1_default_y)
    c1_z = c1_default_z  # always on table — the arm lifts it

    # Clamp cube1 to workspace bounds
    c1_x = c1_x.clamp(WS_X_MIN, WS_X_MAX)
    c1_y = c1_y.clamp(WS_Y_MIN, WS_Y_MAX)

    # After clamping, verify separation is still >= MIN_CUBE_SEPARATION
    # If clamping pushed cubes too close, shift cube1 to maintain min distance
    for _ in range(5):
        dist_c = torch.sqrt((c1_x - c2_x)**2 + (c1_y - c2_y)**2)
        too_close = dist_c < MIN_CUBE_SEPARATION
        if not too_close.any():
            break
        cnt = too_close.sum().item()
        # Re-randomize the angle for too-close cubes
        new_angle = torch.empty(cnt, device=device).uniform_(0, 2 * math.pi)
        c1_x[too_close] = c2_x[too_close] + actual_sep[too_close] * torch.cos(new_angle)
        c1_y[too_close] = c2_y[too_close] + actual_sep[too_close] * torch.sin(new_angle)
        c1_x = c1_x.clamp(WS_X_MIN, WS_X_MAX)
        c1_y = c1_y.clamp(WS_Y_MIN, WS_Y_MAX)

    c1_state = cube1.data.default_root_state[env_ids].clone()
    c1_state[:, 0] = c1_x
    c1_state[:, 1] = c1_y
    c1_state[:, 2] = c1_z
    c1_state[:, 0:3] += env.scene.env_origins[env_ids]
    c1_state[:, 7:13] = 0
    cube1.write_root_pose_to_sim(c1_state[:, :7], env_ids=env_ids)
    cube1.write_root_velocity_to_sim(c1_state[:, 7:13], env_ids=env_ids)

    # --- Robot arm: at d=0, use NEAR_CUBE_JOINTS (arm near table). At d=1, DEFAULT_JOINTS ---
    near_joints = torch.tensor(NEAR_CUBE_JOINTS, device=device, dtype=torch.float32)
    default_joints = torch.tensor(DEFAULT_JOINTS, device=device, dtype=torch.float32)
    target_joints = near_joints.unsqueeze(0).expand(n, -1).clone()
    
    # Interpolate arm pose: d=0 → near cube, d=1 → default retracted
    for j in range(7):
        target_joints[:, j] = near_joints[j] + d_vals * (default_joints[j] - near_joints[j])

    # Add small noise
    noise = torch.randn(n, 7, device=device) * 0.02
    target_joints[:, :7] += noise

    # Clamp to joint limits
    joint_limits = robot.data.soft_joint_pos_limits[env_ids]
    target_joints = target_joints.clamp(joint_limits[..., 0], joint_limits[..., 1])

    # Gripper: closed at d=0 (holding cube), open at d=1
    gripper_pos = d_vals * GRIPPER_OPEN
    target_joints[:, 7] = gripper_pos
    target_joints[:, 8] = gripper_pos

    joint_vel = torch.zeros_like(target_joints)
    robot.set_joint_position_target(target_joints, env_ids=env_ids)
    robot.set_joint_velocity_target(joint_vel, env_ids=env_ids)
    robot.write_joint_state_to_sim(target_joints, joint_vel, env_ids=env_ids)
    
    # DIAGNOSTIC: verify reset state
    if not hasattr(place_reset, '_diag_count'):
        place_reset._diag_count = 0
    if place_reset._diag_count < 2:
        place_reset._diag_count += 1
        i = 0
        print(f"\nRESET DIAG (call {place_reset._diag_count}):")
        print(f"  d_val[0] = {d_vals[i].item():.3f}")
        print(f"  target_joints[0][:4] = {target_joints[i, :4].tolist()}")
        print(f"  gripper[0] = {target_joints[i, 7].item():.4f}, {target_joints[i, 8].item():.4f}")
        print(f"  cube1 pos = ({c1_x[i].item():.3f}, {c1_y[i].item():.3f}, {c1_z[i].item():.3f})")
        print(f"  cube2 pos = ({c2_x[i].item():.3f}, {c2_y[i].item():.3f})")
        print(f"  NEAR_CUBE_JOINTS[:4] = {NEAR_CUBE_JOINTS[:4]}")
        print()


# ─── CONFIG CLASSES ───────────────────────────────────────────

@configclass
class GraspRewardsCfg:
    """PLACE skill: primary place reward + action penalty."""
    grasp = RewTerm(func=place_reward, weight=1.0)
    action_rate = RewTerm(func=action_penalty, weight=-0.05)


@configclass
class GraspCurriculumCfg:
    skill_curriculum = CurrTerm(func=place_curriculum)


@configclass
class GraspEventCfg:
    curriculum_reset = EventTerm(
        func=place_reset,
        mode="reset",
    )
