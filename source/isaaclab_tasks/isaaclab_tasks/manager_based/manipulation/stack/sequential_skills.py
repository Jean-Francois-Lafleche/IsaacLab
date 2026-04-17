# Sequential Skill Training v3: One reward at a time
#
# Paradigm:
#   - Train one skill at a time with a single reward function
#   - Same policy network throughout — checkpoint chains across skills
#   - Per-skill difficulty curriculum (distribution-based)
#   - Grasp validated by sustained 1-second hold at height
#   - Skill sequence: reach -> grasp -> lift -> transport -> place -> stack
#
# Scene constants (from stack_joint_pos_env_cfg.py):
#   cube1 default: (0.4, 0.0, 0.0203)  — 0.0203 is local z, actual root z ~ 0.039
#   cube2 default: (0.55, 0.05, 0.0203)
#   cube height: 0.046m
#   gripper open: 0.04
#   env_step_dt ~ 0.01667s (decimation=2, sim_dt=1/120)

from __future__ import annotations
import json
import math
from typing import TYPE_CHECKING
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import FrameTransformer
from isaaclab.managers import CurriculumTermCfg as CurrTerm, EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp import reset_scene_to_default

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

CONFIG_PATH = "/tmp/llm_trainer_v3_config.json"

# Scene constants
CUBE_INIT_Z = 0.039     # actual root pos z relative to env origin
CUBE_HEIGHT = 0.046
CUBE1_DEFAULT_XY = (0.4, 0.0)
CUBE2_DEFAULT_XY = (0.55, 0.05)
GRIPPER_OPEN = 0.04
ENV_STEP_DT = 0.01667
HOLD_DURATION_SEC = 1.0
HOLD_STEPS = int(HOLD_DURATION_SEC / ENV_STEP_DT)  # ~60 steps

SKILL_SEQUENCE = ["reach", "grasp", "lift", "transport", "place", "release", "stack"]
DEFAULT_ADVANCEMENT_THRESHOLD = 0.90
DEFAULT_ADVANCEMENT_WINDOW = 100


def _load_cfg():
    try:
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
    except Exception:
        return {
            "version": 0,
            "current_skill": "reach",
            "skill_sequence": SKILL_SEQUENCE,
            "advancement_threshold": DEFAULT_ADVANCEMENT_THRESHOLD,
            "advancement_window": DEFAULT_ADVANCEMENT_WINDOW,
            "per_skill_config": {},
            "action_penalty_weight": -0.05,
            "promotion_only_until": 0.3,
        }
    # ── Guard: ensure current_skill is always present and consistent ──
    skill_seq = cfg.get("skill_sequence", SKILL_SEQUENCE)
    if "current_skill" not in cfg:
        # Derive from skill_index if available, otherwise default to first skill
        idx = int(cfg.get("skill_index", 0))
        idx = max(0, min(idx, len(skill_seq) - 1))
        cfg["current_skill"] = skill_seq[idx]
        cfg["_repaired"] = f"current_skill was missing, derived from skill_index={idx}"
        _save_cfg(cfg)
    elif cfg["current_skill"] not in skill_seq:
        # current_skill is set but invalid — clamp to nearest valid
        idx = int(cfg.get("skill_index", 0))
        idx = max(0, min(idx, len(skill_seq) - 1))
        cfg["current_skill"] = skill_seq[idx]
        cfg["_repaired"] = f"current_skill was invalid, reset from skill_index={idx}"
        _save_cfg(cfg)
    # Also keep skill_index in sync with current_skill
    expected_idx = skill_seq.index(cfg["current_skill"]) if cfg["current_skill"] in skill_seq else 0
    if cfg.get("skill_index") is not None and int(cfg["skill_index"]) != expected_idx:
        cfg["skill_index"] = expected_idx
        _save_cfg(cfg)
    return cfg


def _save_cfg(cfg):
    """Write config, ensuring required keys are never dropped."""
    skill_seq = cfg.get("skill_sequence", SKILL_SEQUENCE)
    # Guarantee current_skill is present and valid before every write
    if "current_skill" not in cfg or cfg["current_skill"] not in skill_seq:
        idx = int(cfg.get("skill_index", 0))
        idx = max(0, min(idx, len(skill_seq) - 1))
        cfg["current_skill"] = skill_seq[idx]
    # Keep skill_index in sync
    cfg["skill_index"] = skill_seq.index(cfg["current_skill"])
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# SHARED STATE
# ══════════════════════════════════════════════════════════════════════════════

def _get_state(env):
    """Compute all scene variables once per step."""
    ee: FrameTransformer = env.scene["ee_frame"]
    cube1: RigidObject = env.scene["cube_1"]
    cube2: RigidObject = env.scene["cube_2"]
    robot: Articulation = env.scene["robot"]

    ee_pos = ee.data.target_pos_w[:, 0, :]
    rfinger_pos = ee.data.target_pos_w[:, 1, :]
    lfinger_pos = ee.data.target_pos_w[:, 2, :]
    c1_pos = cube1.data.root_pos_w
    c1_vel = cube1.data.root_lin_vel_w
    c2_pos = cube2.data.root_pos_w
    origins = env.scene.env_origins

    # Distances
    ee_to_c1 = torch.norm(c1_pos - ee_pos, dim=1)
    ee_to_c1_xy = torch.norm(c1_pos[:, :2] - ee_pos[:, :2], dim=1)
    c1_to_c2_xy = torch.norm(c1_pos[:, :2] - c2_pos[:, :2], dim=1)

    # Heights relative to env origin
    c1_h = c1_pos[:, 2] - origins[:, 2]
    c2_h = c2_pos[:, 2] - origins[:, 2]
    c1_above_c2 = c1_pos[:, 2] - c2_pos[:, 2]

    # Cube velocity magnitude
    c1_speed = torch.norm(c1_vel, dim=1)

    # Gripper state
    f1_idx = robot.data.joint_names.index("panda_finger_joint1")
    f2_idx = robot.data.joint_names.index("panda_finger_joint2")
    finger_pos = robot.data.joint_pos[:, [f1_idx, f2_idx]]
    openness = (finger_pos.mean(dim=1) / GRIPPER_OPEN).clamp(0, 1)

    # Fingertip spatial check: cube between fingers in Y
    cube_between_y = (
        ((rfinger_pos[:, 1] < c1_pos[:, 1]) & (c1_pos[:, 1] < lfinger_pos[:, 1])) |
        ((lfinger_pos[:, 1] < c1_pos[:, 1]) & (c1_pos[:, 1] < rfinger_pos[:, 1]))
    )
    z_aligned = torch.abs(c1_pos[:, 2] - ee_pos[:, 2]) < 0.05
    spatially_grasped = cube_between_y & z_aligned & (openness < 0.7)

    # Joint velocity (arm only)
    joint_vel = torch.norm(robot.data.joint_vel[:, :7], dim=1)
    ee_h = ee_pos[:, 2] - origins[:, 2]

    return {
        "ee_pos": ee_pos, "rfinger_pos": rfinger_pos, "lfinger_pos": lfinger_pos,
        "c1_pos": c1_pos, "c2_pos": c2_pos, "origins": origins,
        "ee_to_c1": ee_to_c1, "ee_to_c1_xy": ee_to_c1_xy,
        "c1_to_c2_xy": c1_to_c2_xy,
        "c1_h": c1_h, "c2_h": c2_h, "c1_above_c2": c1_above_c2,
        "c1_speed": c1_speed,
        "openness": openness, "finger_pos": finger_pos,
        "spatially_grasped": spatially_grasped,
        "joint_vel": joint_vel, "ee_h": ee_h,
    }


# ══════════════════════════════════════════════════════════════════════════════
# EPISODE TRACKERS
# ══════════════════════════════════════════════════════════════════════════════

_hold_counter = None  # consecutive steps holding cube steady at height
_last_success = None  # cached success state (evaluated before reset)
_episode_success = None  # did the env succeed at any point during the episode

def _init_trackers(n, device):
    global _hold_counter, _last_success, _episode_success
    _hold_counter = torch.zeros(n, dtype=torch.long, device=device)
    _last_success = torch.zeros(n, dtype=torch.bool, device=device)
    _episode_success = torch.zeros(n, dtype=torch.bool, device=device)

def _reset_trackers(env_ids):
    global _hold_counter, _episode_success
    if _hold_counter is not None:
        _hold_counter[env_ids] = 0
    if _episode_success is not None:
        _episode_success[env_ids] = False

def _update_hold_counter(s):
    """Update the sustained-hold counter for grasp validation."""
    global _hold_counter
    lifted = s["c1_h"] > 0.06  # ~2cm above resting
    steady = s["c1_speed"] < 0.01  # m/s
    holding = lifted & steady & s["spatially_grasped"]

    _hold_counter[holding] += 1
    _hold_counter[~holding] = 0


# ══════════════════════════════════════════════════════════════════════════════
# REWARD FUNCTIONS — one per skill
# ══════════════════════════════════════════════════════════════════════════════

def reach_reward(env) -> torch.Tensor:
    """Dense reach reward: get EE close to cube1.
    t=0: EE ~15cm from cube → ~0.3. ✓
    success: EE at cube → ~1.0. ✓
    """
    s = _get_state(env)
    if _hold_counter is None:
        _init_trackers(env.num_envs, env.device)
    _update_hold_counter(s)
    coarse = 1.0 - torch.tanh(s["ee_to_c1"] / 0.1)
    fine = torch.exp(-(s["ee_to_c1"] ** 2) / (0.02 ** 2))
    return coarse * 0.4 + fine * 0.6


def grasp_reward(env) -> torch.Tensor:
    """Dense grasp reward: approach + close gripper + lift slightly.
    Encourages full grasp-and-hold behavior.
    t=0: EE far, gripper open → ~0.1 (approach component). ✓
    success: cube held at height for 1s → ~1.0. ✓
    gaming: pushing gives transient contact, no sustained hold. ✓
    """
    s = _get_state(env)
    if _hold_counter is None:
        _init_trackers(env.num_envs, env.device)
    _update_hold_counter(s)

    # Component 1: approach (always active, small weight)
    approach = 1.0 - torch.tanh(s["ee_to_c1"] / 0.1)

    # Component 2: gripper closure when near cube
    near = (s["ee_to_c1"] < 0.04).float()
    closure = near * (1.0 - s["openness"])

    # Component 3: spatial grasp posture
    grasp_posture = s["spatially_grasped"].float()

    # Component 4: height when grasped (the real objective)
    height_norm = ((s["c1_h"] - CUBE_INIT_Z) / 0.05).clamp(0, 1)
    lift_while_grasped = grasp_posture * height_norm

    # Component 5: steadiness bonus (hold counter progress toward 1s)
    hold_progress = (_hold_counter.float() / HOLD_STEPS).clamp(0, 1)
    steadiness = grasp_posture * hold_progress

    return approach * 0.1 + closure * 0.15 + grasp_posture * 0.15 + lift_while_grasped * 0.3 + steadiness * 0.3


def lift_reward(env) -> torch.Tensor:
    """Dense lift reward: lift cube1 high while maintaining grasp.
    Gated on spatial grasp to prevent reward from pushing.
    """
    s = _get_state(env)
    if _hold_counter is None:
        _init_trackers(env.num_envs, env.device)
    _update_hold_counter(s)

    grasped = s["spatially_grasped"].float()
    # Normalized height: 0 at rest (0.039), 1 at target (0.12)
    height_norm = ((s["c1_h"] - CUBE_INIT_Z) / 0.08).clamp(0, 1)
    return grasped * height_norm


def transport_reward(env) -> torch.Tensor:
    """Dense transport reward: move cube1 above cube2 while holding it.
    Gated on cube1 being lifted (height > 0.08).
    """
    s = _get_state(env)
    if _hold_counter is None:
        _init_trackers(env.num_envs, env.device)
    _update_hold_counter(s)

    lifted = (s["c1_h"] > 0.08).float()
    grasped = s["spatially_grasped"].float()
    proximity = 1.0 - torch.tanh(s["c1_to_c2_xy"] / 0.08)
    return lifted * grasped * proximity


def place_reward(env) -> torch.Tensor:
    """Dense place reward: pick cube1, bring it above cube2, and release.

    Three components for smooth gradient from any starting state:
    1. Approach shaping: reward proximity of cube1 to above-cube2 target position (always active)
    2. Alignment bonus: being directly above cube2 at correct height
    3. Release bonus: opening gripper while aligned (dominant signal)
    """
    s = _get_state(env)
    if _hold_counter is None:
        _init_trackers(env.num_envs, env.device)
    _update_hold_counter(s)

    # Target position: directly above cube2 at stacking height
    # c1_to_c2_xy: horizontal distance between cubes
    # c1_above_c2: height of cube1 above cube2
    target_height = CUBE_HEIGHT  # 0.046m above cube2

    # Component 1: Approach shaping — dense reward for getting close to target
    # Combines XY approach + height approach into a single distance metric
    xy_dist = s["c1_to_c2_xy"]
    height_error = torch.abs(s["c1_above_c2"] - target_height)
    # Must be lifted first — no reward for sliding on table
    lifted = (s["c1_h"] > CUBE_INIT_Z + 0.02).float()  # at least 2cm above resting
    approach = torch.exp(-10.0 * xy_dist) * torch.exp(-30.0 * height_error) * lifted
    approach_reward = approach * 0.2

    # Component 2: Alignment — tight check that cube1 is in place position
    aligned = (xy_dist < 0.06).float()
    height_match = torch.exp(-((s["c1_above_c2"] - target_height) ** 2) / (0.015 ** 2))
    alignment_reward = aligned * height_match * lifted * 0.1

    # Component 3: Release — dominant signal, linear in openness for strong gradient
    positioned = aligned * (height_match > 0.5).float() * lifted
    release_reward = positioned * s["openness"] * 0.7

    return approach_reward + alignment_reward + release_reward


def release_reward(env) -> torch.Tensor:
    """Dense release reward: open gripper, retract arm, stop moving.

    Assumes cube1 is already placed on cube2 (stacked). The agent must:
    1. Open gripper fully (release the cube)
    2. Retract EE away from the stack (move up/back)
    3. Come to a rest (low joint velocity)

    Three components:
    - Openness: linear in gripper openness (strong gradient to open)
    - Retraction: reward EE moving away from cube1 (vertical + horizontal)
    - Stillness: exponential bonus for low joint velocity
    """
    s = _get_state(env)
    if _hold_counter is None:
        _init_trackers(env.num_envs, env.device)
    _update_hold_counter(s)

    # Component 1: Open gripper — linear for strong gradient
    open_reward = s["openness"].clamp(0, 1) * 0.3

    # Component 2: Retract EE away from cube stack
    # Target: EE should be ≥10cm above table and ≥8cm horizontally from cube1
    ee_above_table = (s["ee_h"] - CUBE_INIT_Z) / 0.15  # normalized: 0 at table, 1 at 15cm above
    ee_above_table = ee_above_table.clamp(0, 1)
    ee_horiz_dist = s["ee_to_c1_xy"] / 0.10  # normalized: 1 at 10cm away
    ee_horiz_dist = ee_horiz_dist.clamp(0, 1)
    retract_reward = (ee_above_table * 0.5 + ee_horiz_dist * 0.5) * 0.3

    # Component 3: Stillness — exponential decay on joint velocity
    # Only give this when gripper is mostly open (don't reward freezing while gripping)
    mostly_open = (s["openness"] > 0.6).float()
    stillness = torch.exp(-(s["joint_vel"] ** 2) / (0.3 ** 2))
    still_reward = mostly_open * stillness * 0.4

    return open_reward + retract_reward + still_reward


def stack_reward(env) -> torch.Tensor:
    """Dense stack reward: full task completion.
    Cube1 stacked on cube2, released, robot at rest.
    """
    s = _get_state(env)
    if _hold_counter is None:
        _init_trackers(env.num_envs, env.device)
    _update_hold_counter(s)

    aligned = torch.exp(-(s["c1_to_c2_xy"] ** 2) / (0.03 ** 2))
    correct_height = torch.exp(-((s["c1_above_c2"] - CUBE_HEIGHT) ** 2) / (0.01 ** 2))
    released = s["openness"].clamp(0, 1)
    robot_calm = torch.exp(-(s["joint_vel"] ** 2) / (0.5 ** 2))
    return aligned * correct_height * released * robot_calm


# ══════════════════════════════════════════════════════════════════════════════
# SUCCESS CHECKS — one per skill (binary, for advancement gating)
# ══════════════════════════════════════════════════════════════════════════════

def reach_success(env, s) -> torch.Tensor:
    return (s["ee_to_c1"] < 0.03).float()

def grasp_success(env, s) -> torch.Tensor:
    """Grasp success: sustained hold for >= HOLD_STEPS consecutive steps."""
    return (_hold_counter >= HOLD_STEPS).float()

def lift_success(env, s) -> torch.Tensor:
    return ((s["c1_h"] > 0.12) & (s["c1_speed"] < 0.02) & s["spatially_grasped"]).float()

def transport_success(env, s) -> torch.Tensor:
    return ((s["c1_to_c2_xy"] < 0.04) & (s["c1_above_c2"] > 0.02) & (s["c1_h"] > 0.08)).float()

def place_success(env, s) -> torch.Tensor:
    aligned = s["c1_to_c2_xy"] < 0.04
    correct_h = torch.abs(s["c1_above_c2"] - CUBE_HEIGHT) < 0.015
    released = s["openness"] > 0.75
    return (aligned & correct_h & released).float()

def release_success(env, s) -> torch.Tensor:
    """Release success: gripper open, EE retracted, robot stopped, cube still stacked."""
    gripper_open = s["openness"] > 0.75
    ee_retracted = s["ee_h"] > (CUBE_INIT_Z + 0.08)  # at least 8cm above table
    ee_away = s["ee_to_c1_xy"] > 0.05  # at least 5cm horizontally from cube
    robot_stopped = s["joint_vel"] < 0.3
    # Cube must still be stacked (didn't knock it off)
    still_stacked = (s["c1_to_c2_xy"] < 0.05) & (torch.abs(s["c1_above_c2"] - CUBE_HEIGHT) < 0.03)
    return (gripper_open & ee_retracted & ee_away & robot_stopped & still_stacked).float()

def stack_success(env, s) -> torch.Tensor:
    aligned = s["c1_to_c2_xy"] < 0.05
    correct_h = torch.abs(s["c1_above_c2"] - CUBE_HEIGHT) < 0.02
    above = s["c1_above_c2"] > 0
    released = s["openness"] > 0.75
    robot_stopped = s["joint_vel"] < 0.5
    return (aligned & correct_h & above & released & robot_stopped).float()

SUCCESS_FNS = {
    "reach": reach_success,
    "grasp": grasp_success,
    "lift": lift_success,
    "transport": transport_success,
    "place": place_success,
    "release": release_success,
    "stack": stack_success,
}

REWARD_FNS = {
    "reach": reach_reward,
    "grasp": grasp_reward,
    "lift": lift_reward,
    "transport": transport_reward,
    "place": place_reward,
    "release": release_reward,
    "stack": stack_reward,
}


# ══════════════════════════════════════════════════════════════════════════════
# SKILL ROUTER — the single active reward
# ══════════════════════════════════════════════════════════════════════════════

def skill_router_reward(env) -> torch.Tensor:
    """Routes to the currently active skill's reward function.
    This is the ONLY reward in the config (besides action_penalty).
    """
    cfg = _load_cfg()
    current_skill = cfg.get("current_skill", "reach")
    reward_fn = REWARD_FNS.get(current_skill, reach_reward)
    return reward_fn(env)


def action_penalty(env) -> torch.Tensor:
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


# ══════════════════════════════════════════════════════════════════════════════
# CURRICULUM: per-skill difficulty + skill advancement
# ══════════════════════════════════════════════════════════════════════════════

def sequential_curriculum(env, env_ids):
    """Curriculum with per-skill difficulty and automatic skill advancement."""
    global _episode_success, _hold_counter
    if not hasattr(sequential_curriculum, "_n"):
        sequential_curriculum._n = 0
        sequential_curriculum._difficulty = torch.zeros(env.num_envs, device=env.device)
        sequential_curriculum._success_ema = 0.0
        sequential_curriculum._success_window = []
        sequential_curriculum._current_skill = None
    sequential_curriculum._n += 1

    cfg = _load_cfg()
    current_skill = cfg.get("current_skill", "reach")
    skill_seq = cfg.get("skill_sequence", SKILL_SEQUENCE)
    threshold = cfg.get("advancement_threshold", DEFAULT_ADVANCEMENT_THRESHOLD)
    window_size = cfg.get("advancement_window", DEFAULT_ADVANCEMENT_WINDOW)
    d = sequential_curriculum._difficulty

    # Detect skill transition — reset difficulty
    if sequential_curriculum._current_skill != current_skill:
        sequential_curriculum._current_skill = current_skill
        sequential_curriculum._difficulty.zero_()
        sequential_curriculum._success_window = []
        sequential_curriculum._success_ema = 0.0
        d = sequential_curriculum._difficulty

    if isinstance(env_ids, torch.Tensor):
        ids = env_ids
    else:
        ids = torch.tensor(env_ids, device=env.device, dtype=torch.long) if not isinstance(env_ids, slice) else torch.arange(env.num_envs, device=env.device)

    # Compute success for current skill
    s = _get_state(env)
    if _hold_counter is None:
        _init_trackers(env.num_envs, env.device)
    _update_hold_counter(s)

    success_fn = SUCCESS_FNS.get(current_skill, reach_success)
    instant_success = success_fn(env, s)
    # Track per-episode success: True if env succeeded at any point this episode
    if _episode_success is not None:
        _episode_success |= instant_success.bool()
    mean_success = instant_success.mean().item()

    # Update success tracking
    sequential_curriculum._success_window.append(mean_success)
    if len(sequential_curriculum._success_window) > window_size:
        sequential_curriculum._success_window = sequential_curriculum._success_window[-window_size:]
    ema_alpha = 0.02
    sequential_curriculum._success_ema = (
        (1 - ema_alpha) * sequential_curriculum._success_ema + ema_alpha * mean_success
    )

    # Advance difficulty per-env
    if ids.numel() > 0 and sequential_curriculum._n > 2:
        skill_cfg = cfg.get("per_skill_config", {}).get(current_skill, {})
        step_up = skill_cfg.get("step_up", 0.02)
        step_down = skill_cfg.get("step_down", 0.005)
        promotion_only_until = cfg.get("promotion_only_until", 0.3)

        # Use episode success (was this env successful at any point before reset?)
        # NOT instantaneous success (which is meaningless at reset time)
        ep_success = _episode_success[ids] if _episode_success is not None else instant_success[ids].bool()
        promotion_only = d[ids] < promotion_only_until
        demote = torch.where(promotion_only, torch.zeros_like(d[ids]), torch.full_like(d[ids], step_down))
        delta = torch.where(ep_success, step_up, -demote)
        d[ids] = (d[ids] + delta).clamp(0.0, 1.0)

        # Panic threshold
        if sequential_curriculum._n > 200 and sequential_curriculum._success_ema < 0.10:
            d.clamp_(max=d.mean().item() - 0.05)
            d.clamp_(min=0.0)

    # Check for skill advancement
    advanced = False
    if len(sequential_curriculum._success_window) >= window_size:
        window_mean = sum(sequential_curriculum._success_window[-window_size:]) / window_size
        mean_d = d.mean().item()
        if window_mean >= threshold and mean_d > 0.7:
            # Advance to next skill
            skill_idx = skill_seq.index(current_skill) if current_skill in skill_seq else 0
            if skill_idx + 1 < len(skill_seq):
                next_skill = skill_seq[skill_idx + 1]
                cfg["current_skill"] = next_skill
                cfg["version"] = cfg.get("version", 0) + 1
                cfg["notes"] = f"Auto-advanced from {current_skill} to {next_skill} at iter {sequential_curriculum._n}"
                _save_cfg(cfg)
                advanced = True

    # Return metrics (all values must be numeric — logger can't handle strings)
    mean_d = d.mean().item()
    skill_idx = skill_seq.index(current_skill) if current_skill in skill_seq else 0
    return {
        "skill_index": float(skill_idx),
        "skill_reward": mean_success,
        "success_ema": sequential_curriculum._success_ema,
        "mean_difficulty": mean_d,
        "d10": torch.quantile(d, 0.1).item() if d.numel() > 0 else 0,
        "d90": torch.quantile(d, 0.9).item() if d.numel() > 0 else 0,
        "hold_counter_mean": _hold_counter.float().mean().item() if _hold_counter is not None else 0,
        "advanced": 1.0 if advanced else 0.0,
        "config_version": float(cfg.get("version", 0)),
        "iter": float(sequential_curriculum._n),
    }


# ══════════════════════════════════════════════════════════════════════════════
# RESET: per-skill spawn conditions
# ══════════════════════════════════════════════════════════════════════════════

def sequential_reset(env, env_ids, spawn_range=0.02):
    """Reset with per-skill difficulty-based spawning."""
    reset_scene_to_default(env, env_ids)
    _reset_trackers(env_ids)

    cfg = _load_cfg()
    current_skill = cfg.get("current_skill", "reach")
    sr = cfg.get("spawn_range", spawn_range)

    cube1: RigidObject = env.scene["cube_1"]
    cube2: RigidObject = env.scene["cube_2"]
    robot: Articulation = env.scene["robot"]
    n = len(env_ids)

    tracker = getattr(sequential_curriculum, "_difficulty", None)
    if tracker is None:
        return
    d = tracker[env_ids]

    c1 = cube1.data.default_root_state[env_ids].clone()
    c2 = cube2.data.default_root_state[env_ids].clone()

    if current_skill == "reach":
        # d=0: cube1 very close to EE start position (~0.4, 0.0)
        # d=1: cube1 at full workspace range
        # EE rests around x≈0.45, y≈0.0 at default joint pose
        ee_rest_x, ee_rest_y = 0.45, 0.0
        for i in range(n):
            di = d[i].item()
            max_offset = 0.02 + di * 0.18  # d=0: ±2cm, d=1: ±20cm
            c1[i, 0] = ee_rest_x + torch.empty(1).uniform_(-max_offset, max_offset).item()
            c1[i, 1] = ee_rest_y + torch.empty(1).uniform_(-max_offset, max_offset).item()
            # Clamp to workspace
            c1[i, 0] = max(0.25, min(0.65, c1[i, 0].item()))
            c1[i, 1] = max(-0.2, min(0.2, c1[i, 1].item()))

    elif current_skill == "grasp":
        # d=0: cube1 nearly in-hand (very close to EE, ~1-2cm)
        # d=1: cube1 at normal position requiring full approach
        ee_rest_x, ee_rest_y = 0.45, 0.0
        for i in range(n):
            di = d[i].item()
            # At d=0: 1cm from EE. At d=1: up to 20cm
            max_dist = 0.01 + di * 0.19
            dist = torch.empty(1).uniform_(0.01, max(0.011, max_dist)).item()
            angle = torch.empty(1).uniform_(0, 2 * math.pi).item()
            c1[i, 0] = ee_rest_x + dist * math.cos(angle)
            c1[i, 1] = ee_rest_y + dist * math.sin(angle)
            c1[i, 0] = max(0.25, min(0.65, c1[i, 0].item()))
            c1[i, 1] = max(-0.2, min(0.2, c1[i, 1].item()))

    elif current_skill == "lift":
        # d=0: cube1 near EE (easy to grab and lift a bit)
        # d=1: cube1 further, needs full approach + grasp + high lift
        for i in range(n):
            di = d[i].item()
            max_offset = 0.02 + di * 0.15
            c1[i, 0] = 0.45 + torch.empty(1).uniform_(-max_offset, max_offset).item()
            c1[i, 1] = 0.0 + torch.empty(1).uniform_(-max_offset, max_offset).item()
            c1[i, 0] = max(0.25, min(0.65, c1[i, 0].item()))
            c1[i, 1] = max(-0.2, min(0.2, c1[i, 1].item()))

    elif current_skill == "release":
        # d=0: cube already stacked, gripper open, EE near retracted position
        # d=1: cube stacked but gripper closed and EE still near cube (full release task)
        for i in range(n):
            di = d[i].item()
            # Cube2 at a fixed reasonable position
            c2[i, 0] = 0.45 + torch.empty(1).uniform_(-sr, sr).item()
            c2[i, 1] = 0.0 + torch.empty(1).uniform_(-sr, sr).item()
            # Cube1 stacked on cube2
            c1[i, 0] = c2[i, 0].item() + torch.empty(1).uniform_(-0.01, 0.01).item()
            c1[i, 1] = c2[i, 1].item() + torch.empty(1).uniform_(-0.01, 0.01).item()
            c1[i, 2] = c2[i, 2].item() + CUBE_HEIGHT  # stacked height

        # Write cubes early so robot can be positioned relative to stack
        c1_write = c1.clone()
        c1_write[:, 0:3] += env.scene.env_origins[env_ids]
        c1_write[:, 7:13] = 0
        cube1.write_root_pose_to_sim(c1_write[:, :7], env_ids=env_ids)
        cube1.write_root_velocity_to_sim(c1_write[:, 7:13], env_ids=env_ids)
        c2_write = c2.clone()
        c2_write[:, 0:3] += env.scene.env_origins[env_ids]
        c2_write[:, 7:13] = 0
        cube2.write_root_pose_to_sim(c2_write[:, :7], env_ids=env_ids)
        cube2.write_root_velocity_to_sim(c2_write[:, 7:13], env_ids=env_ids)

        # At d=0, set gripper open; at d=1, gripper closed (needs to learn to open)
        # Robot joint positions: adjust gripper fingers based on difficulty
        if hasattr(robot.data, 'default_joint_pos'):
            jpos = robot.data.default_joint_pos[env_ids].clone()
            f1_idx = robot.data.joint_names.index("panda_finger_joint1")
            f2_idx = robot.data.joint_names.index("panda_finger_joint2")
            for i in range(n):
                di = d[i].item()
                # d=0: gripper fully open (0.04), d=1: gripper closed (0.0)
                finger_val = GRIPPER_OPEN * (1.0 - di)
                jpos[i, f1_idx] = finger_val
                jpos[i, f2_idx] = finger_val
            robot.write_joint_state_to_sim(jpos, robot.data.default_joint_vel[env_ids], env_ids=env_ids)
        return  # already wrote cubes above

    elif current_skill in ("transport", "place", "stack"):
        # d=0: cubes very close together
        # d=1: cubes at full separation
        for i in range(n):
            di = d[i].item()
            # Cube1 at slight random offset
            c1[i, 0] = 0.4 + torch.empty(1).uniform_(-sr, sr).item()
            c1[i, 1] = 0.0 + torch.empty(1).uniform_(-sr, sr).item()
            # Cube2 distance scales with difficulty
            min_sep = 0.05
            max_sep = min_sep + di * 0.13  # d=0: 5cm, d=1: 18cm
            actual_sep = torch.empty(1).uniform_(min_sep, max(min_sep + 0.001, max_sep)).item()
            angle = torch.empty(1).uniform_(0, 2 * math.pi).item()
            c2[i, 0] = c1[i, 0].item() + actual_sep * math.cos(angle)
            c2[i, 1] = c1[i, 1].item() + actual_sep * math.sin(angle)
            c2[i, 0] = max(0.25, min(0.65, c2[i, 0].item()))
            c2[i, 1] = max(-0.2, min(0.2, c2[i, 1].item()))
    else:
        # Default: small random offset
        for i in range(n):
            c1[i, 0] += torch.empty(1).uniform_(-sr, sr).item()
            c1[i, 1] += torch.empty(1).uniform_(-sr, sr).item()

    # Write cube1
    c1[:, 0:3] += env.scene.env_origins[env_ids]
    c1[:, 7:13] = 0
    cube1.write_root_pose_to_sim(c1[:, :7], env_ids=env_ids)
    cube1.write_root_velocity_to_sim(c1[:, 7:13], env_ids=env_ids)

    # Write cube2
    c2[:, 0:3] += env.scene.env_origins[env_ids]
    c2[:, 7:13] = 0
    cube2.write_root_pose_to_sim(c2[:, :7], env_ids=env_ids)
    cube2.write_root_velocity_to_sim(c2[:, 7:13], env_ids=env_ids)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@configclass
class SequentialRewardsCfg:
    skill = RewTerm(func=skill_router_reward, weight=1.0)
    action_rate = RewTerm(func=action_penalty, weight=-0.05)

@configclass
class SequentialCurriculumCfg:
    sequential = CurrTerm(func=sequential_curriculum)

@configclass
class SequentialEventCfg:
    reset = EventTerm(func=sequential_reset, mode="reset", params={"spawn_range": 0.02})
