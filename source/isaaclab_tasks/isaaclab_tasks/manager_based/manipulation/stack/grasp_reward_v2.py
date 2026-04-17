# LLM-Tutor v3: Sequential Skill Training with Per-Skill Curriculum
#
# Each skill phase has:
#   - Active rewards (current skill + maintenance of previous skills)
#   - Skill-specific curriculum parameter
#   - Advancement threshold (checked every 100 iters by LLM)
#
# Phase progression stored in config JSON, hot-patchable.

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

CUBE_INIT_Z = 0.039  # actual root pos relative to env_origins
CUBE_HEIGHT = 0.046
GRIPPER_OPEN = 0.04
MIN_C1C2_DIST = 0.05
DEFAULT_C1C2_DIST = 0.158

def _load_cfg():
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# STATE + TRACKERS
# ══════════════════════════════════════════════════════════════════════════════

def _get_state(env):
    ee: FrameTransformer = env.scene["ee_frame"]
    cube1: RigidObject = env.scene["cube_1"]
    cube2: RigidObject = env.scene["cube_2"]
    robot: Articulation = env.scene["robot"]

    ee_pos = ee.data.target_pos_w[:, 0, :]
    rfinger = ee.data.target_pos_w[:, 1, :]
    lfinger = ee.data.target_pos_w[:, 2, :]
    c1_pos = cube1.data.root_pos_w
    c2_pos = cube2.data.root_pos_w
    origins = env.scene.env_origins

    f1_idx = robot.data.joint_names.index("panda_finger_joint1")
    f2_idx = robot.data.joint_names.index("panda_finger_joint2")
    finger_pos = robot.data.joint_pos[:, [f1_idx, f2_idx]]
    openness = (finger_pos.mean(dim=1) / GRIPPER_OPEN).clamp(0, 1)

    # Fingertip spatial grasp check
    cube_between_y = ((rfinger[:, 1] < c1_pos[:, 1]) & (c1_pos[:, 1] < lfinger[:, 1])) | \
                     ((lfinger[:, 1] < c1_pos[:, 1]) & (c1_pos[:, 1] < rfinger[:, 1]))
    z_aligned = torch.abs(c1_pos[:, 2] - ee_pos[:, 2]) < 0.05
    grasped_now = cube_between_y & z_aligned & (openness < 0.7)

    return {
        "ee_pos": ee_pos, "rfinger": rfinger, "lfinger": lfinger,
        "c1_pos": c1_pos, "c2_pos": c2_pos,
        "ee_to_c1": torch.norm(c1_pos - ee_pos, dim=1),
        "c1_to_c2_xy": torch.norm(c1_pos[:, :2] - c2_pos[:, :2], dim=1),
        "c1_h": c1_pos[:, 2] - origins[:, 2],
        "c1_above_c2": c1_pos[:, 2] - c2_pos[:, 2],
        "openness": openness,
        "grasped_now": grasped_now,
    }


# ══════════════════════════════════════════════════════════════════════════════
# REWARD FUNCTIONS — each skill is a separate reward
# ══════════════════════════════════════════════════════════════════════════════

def skill_reach(env, std: float = 0.08) -> torch.Tensor:
    """Reach toward cube1."""
    s = _get_state(env)
    return 1.0 - torch.tanh(s["ee_to_c1"] / std)

def skill_grasp(env) -> torch.Tensor:
    """Fingertip spatial grasp: cube between fingers + Z-aligned."""
    s = _get_state(env)
    # Dense: proximity * closedness when near
    proximity = torch.exp(-(s["ee_to_c1"] ** 2) / (0.03 ** 2))
    closedness = (1.0 - s["openness"]).clamp(0, 1)
    # Binary bonus for actual grasp
    return proximity * closedness * 0.4 + s["grasped_now"].float() * 0.6

def skill_lift(env) -> torch.Tensor:
    """Lift: cube height above table, gated on fingertip grasp."""
    s = _get_state(env)
    height = ((s["c1_h"] - 0.042) / 0.03).clamp(0, 1)
    return s["grasped_now"].float() * height

def skill_transport(env) -> torch.Tensor:
    """Transport: cube near cube2 horizontally, must be lifted + grasped."""
    s = _get_state(env)
    lifted = s["c1_h"] > 0.06
    proximity = 1.0 - torch.tanh(s["c1_to_c2_xy"] / 0.08)
    return (s["grasped_now"] & lifted).float() * proximity

def skill_place(env) -> torch.Tensor:
    """Place: descend toward correct stacking height + open gripper.
    Two components:
    1. Height targeting: reward cube being at stacking height (cube2_z + cube_height)
    2. Release: open gripper when at correct height
    """
    s = _get_state(env)
    above_xy = s["c1_to_c2_xy"] < 0.06
    lifted = s["c1_h"] > 0.06
    
    # Target height: cube2 z + cube height (where cube1 should rest)
    target_h = s["c2_pos"][:, 2] - env.scene.env_origins[:, 2] + CUBE_HEIGHT
    height_error = torch.abs(s["c1_h"] - target_h)
    height_proximity = torch.exp(-(height_error ** 2) / (0.04 ** 2))  # wider gaussian for discoverability
    
    # Dense descent: reward getting closer to target height while above cube2
    descend = (above_xy & lifted).float() * height_proximity * 0.5
    # Release at correct height
    release = (above_xy & lifted).float() * height_proximity * s["openness"] ** 2 * 0.5
    
    return descend + release

def skill_stack(env) -> torch.Tensor:
    """Stack success: correct height + aligned + released."""
    s = _get_state(env)
    aligned = s["c1_to_c2_xy"] < 0.05
    correct_h = torch.abs(s["c1_above_c2"] - CUBE_HEIGHT) < 0.02
    above = s["c1_above_c2"] > 0
    released = s["openness"] > 0.75
    return (aligned & correct_h & above & released).float()

def action_penalty(env) -> torch.Tensor:
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


# ══════════════════════════════════════════════════════════════════════════════
# CURRICULUM — sequential skill phases with per-skill parameters
# ══════════════════════════════════════════════════════════════════════════════

def sequential_curriculum(env, env_ids):
    if not hasattr(sequential_curriculum, "_n"):
        sequential_curriculum._n = 0
        sequential_curriculum._difficulty = torch.zeros(env.num_envs, device=env.device)
        sequential_curriculum._skill_index = 0
    sequential_curriculum._n += 1

    cfg = _load_cfg()
    d = sequential_curriculum._difficulty
    skill_idx = cfg.get("skill_index", sequential_curriculum._skill_index)
    sequential_curriculum._skill_index = skill_idx

    if isinstance(env_ids, torch.Tensor):
        ids = env_ids
    else:
        ids = torch.tensor(env_ids, device=env.device, dtype=torch.long) if not isinstance(env_ids, slice) else torch.arange(env.num_envs, device=env.device)

    # Per-skill advancement
    if ids.numel() > 0 and sequential_curriculum._n > 2:
        with torch.no_grad():
            s = _get_state(env)
            if skill_idx == 0:  # REACH
                success = (s["ee_to_c1"] < 0.05).float()
            elif skill_idx == 1:  # GRASP
                success = s["grasped_now"].float()
            elif skill_idx == 2:  # LIFT
                success = (s["grasped_now"] & (s["c1_h"] > 0.06)).float()
            elif skill_idx == 3:  # TRANSPORT
                success = (s["grasped_now"] & (s["c1_h"] > 0.06) & (s["c1_to_c2_xy"] < 0.06)).float()
            elif skill_idx >= 4:  # PLACE+STACK
                success = skill_stack(env)
            else:
                success = torch.zeros(env.num_envs, device=env.device)

        step_up = cfg.get("step_up", 0.03)
        step_down = cfg.get("step_down", 0.005)
        delta = torch.where(success[ids].bool(), step_up, -torch.full_like(d[ids], step_down))
        d[ids] = (d[ids] + delta).clamp(0.0, 1.0)

    # Hot-patch weights based on current skill phase
    if sequential_curriculum._n % 50 == 0:
        weights = cfg.get("reward_weights", {})
        for name, w in weights.items():
            try:
                idx = env.reward_manager._term_names.index(name)
                env.reward_manager._term_cfgs[idx].weight = w
            except (ValueError, IndexError):
                pass

    mean_d = d.mean().item()
    with torch.no_grad():
        s = _get_state(env)
        return {
            "skill_index": float(skill_idx),
            "skill_reach": skill_reach(env).mean().item(),
            "skill_grasp": skill_grasp(env).mean().item(),
            "skill_lift": skill_lift(env).mean().item(),
            "skill_transport": skill_transport(env).mean().item(),
            "skill_place": skill_place(env).mean().item(),
            "skill_stack": skill_stack(env).mean().item(),
            "pct_grasped": s["grasped_now"].float().mean().item(),
            "pct_lifted": (s["grasped_now"] & (s["c1_h"] > 0.06)).float().mean().item(),
            "mean_difficulty": mean_d,
            "d90": torch.quantile(d, 0.9).item(),
        }


def sequential_reset(env, env_ids, spawn_range=0.02):
    reset_scene_to_default(env, env_ids)
    cfg = _load_cfg()
    sr = cfg.get("spawn_range", spawn_range)
    skill_idx = cfg.get("skill_index", 0)

    cube1: RigidObject = env.scene["cube_1"]
    cube2: RigidObject = env.scene["cube_2"]
    n = len(env_ids)

    c1 = cube1.data.default_root_state[env_ids].clone()
    for i in range(n):
        c1[i, 0] += torch.empty(1).uniform_(-sr, sr).item()
        c1[i, 1] += torch.empty(1).uniform_(-sr, sr).item()
    c1[:, 0:3] += env.scene.env_origins[env_ids]
    c1[:, 7:13] = 0
    cube1.write_root_pose_to_sim(c1[:, :7], env_ids=env_ids)
    cube1.write_root_velocity_to_sim(c1[:, 7:13], env_ids=env_ids)

    # Cube2 positioning depends on skill phase
    c2 = cube2.data.default_root_state[env_ids].clone()
    tracker = getattr(sequential_curriculum, "_difficulty", None)
    if tracker is not None and skill_idx >= 3:
        # Only vary cube2 distance during transport+ phases
        d = tracker[env_ids]
        c1_local = c1[:, :3] - env.scene.env_origins[env_ids]
        c2_default = c2[:, :3].clone()
        for i in range(n):
            di = d[i].item()
            dx = c2_default[i, 0] - c1_local[i, 0]
            dy = c2_default[i, 1] - c1_local[i, 1]
            full_dist = max((dx**2 + dy**2)**0.5, 1e-6)
            lower = MIN_C1C2_DIST + di * 0.1 * full_dist
            upper = MIN_C1C2_DIST + di * (full_dist - MIN_C1C2_DIST)
            lower = min(lower, upper)
            actual = lower + torch.empty(1).uniform_(0, 1).item() * (upper - lower)
            c2[i, 0] = c1_local[i, 0] + actual * dx / full_dist
            c2[i, 1] = c1_local[i, 1] + actual * dy / full_dist
            jitter = 0.01 * max(di, 0.1)
            c2[i, 0] += torch.empty(1).uniform_(-jitter, jitter).item()
            c2[i, 1] += torch.empty(1).uniform_(-jitter, jitter).item()
    elif skill_idx < 3:
        # During reach/grasp/lift: cube2 close to cube1 (easy stacking if attempted)
        c1_local = c1[:, :3] - env.scene.env_origins[env_ids]
        c2_default = c2[:, :3].clone()
        for i in range(n):
            dx = c2_default[i, 0] - c1_local[i, 0]
            dy = c2_default[i, 1] - c1_local[i, 1]
            full_dist = max((dx**2 + dy**2)**0.5, 1e-6)
            c2[i, 0] = c1_local[i, 0] + MIN_C1C2_DIST * dx / full_dist
            c2[i, 1] = c1_local[i, 1] + MIN_C1C2_DIST * dy / full_dist

    c2[:, 0:3] += env.scene.env_origins[env_ids]
    c2[:, 7:13] = 0
    cube2.write_root_pose_to_sim(c2[:, :7], env_ids=env_ids)
    cube2.write_root_velocity_to_sim(c2[:, 7:13], env_ids=env_ids)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@configclass
class GraspV2RewardsCfg:
    """All skill rewards active — weights controlled by curriculum config."""
    skill_reach = RewTerm(func=skill_reach, params={"std": 0.08}, weight=1.5)
    skill_grasp = RewTerm(func=skill_grasp, weight=3.0)
    skill_lift = RewTerm(func=skill_lift, weight=8.0)
    skill_transport = RewTerm(func=skill_transport, weight=5.0)
    skill_place = RewTerm(func=skill_place, weight=8.0)
    skill_stack = RewTerm(func=skill_stack, weight=25.0)
    action_rate = RewTerm(func=action_penalty, weight=-0.1)

@configclass
class GraspV2CurriculumCfg:
    grasp = CurrTerm(func=sequential_curriculum)

@configclass
class GraspV2EventCfg:
    reset = EventTerm(func=sequential_reset, mode="reset", params={"spawn_range": 0.02})
