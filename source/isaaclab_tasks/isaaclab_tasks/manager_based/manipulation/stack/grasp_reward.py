# Grasping-Specific Reward + Curriculum for Franka Stack
#
# Designed specifically for the Franka Panda parallel-jaw gripper:
# - 2 prismatic finger joints: panda_finger_joint1, panda_finger_joint2
# - Binary action: open (0.04) / close (0.0)
# - Finger joint 2 is mirrored (both move inward to close)
#
# GRASPING REWARD DECOMPOSITION:
# 1. EE near object (dense, continuous) — guides approach
# 2. Gripper closing when near object — guides finger action  
# 3. Object between fingers (contact proxy) — guides precise positioning
# 4. Object grasped (lifted) — ultimate grasping success

from __future__ import annotations
import json, math
from collections.abc import Sequence
from typing import TYPE_CHECKING
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import FrameTransformer
from isaaclab.managers import CurriculumTermCfg as CurrTerm, EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp import reset_scene_to_default

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

CONFIG_PATH = "/tmp/llm_trainer_config.json"

def _load_cfg():
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f)
    except:
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# Helper: get finger joint state
# ══════════════════════════════════════════════════════════════════════════════

def _get_finger_state(env):
    """Returns (finger_openness, ee_pos, obj_pos) for all envs.
    finger_openness: 0=closed, 1=open"""
    robot: Articulation = env.scene["robot"]
    gripper_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    finger1 = robot.data.joint_pos[:, gripper_ids[0]]  # 0=closed, 0.04=open
    finger2 = robot.data.joint_pos[:, gripper_ids[1]]  # same (mirrored in hardware)
    # Normalize: 0=closed, 1=open
    open_val = env.cfg.gripper_open_val
    openness = ((finger1 + finger2) / 2.0) / open_val  # avg openness, 0-1
    
    ee: FrameTransformer = env.scene["ee_frame"]
    ee_pos = ee.data.target_pos_w[..., 0, :]
    
    obj: RigidObject = env.scene["cube_1"]
    obj_pos = obj.data.root_pos_w
    
    return openness, ee_pos, obj_pos


# ══════════════════════════════════════════════════════════════════════════════
# Reward functions — embodiment-specific for Franka parallel jaw
# ══════════════════════════════════════════════════════════════════════════════

def approach_reward(env, std=0.08):
    """Dense: distance from EE to cube1. Guides the arm toward the object."""
    _, ee_pos, obj_pos = _get_finger_state(env)
    dist = torch.norm(ee_pos - obj_pos, dim=1)
    return 1 - torch.tanh(dist / std)


def pre_grasp_reward(env):
    """Dense: reward gripper being OPEN when approaching, then CLOSING when very close.
    
    This shapes the natural grasping motion: approach with open gripper,
    then close when the object is between the fingers.
    
    > 5cm from object: reward being OPEN (preparing to grasp)
    < 5cm from object: reward being CLOSED (executing grasp)
    """
    openness, ee_pos, obj_pos = _get_finger_state(env)
    dist = torch.norm(ee_pos - obj_pos, dim=1)
    
    # Far from object: reward open gripper (approach phase)
    far = dist > 0.05
    # Close to object: reward closed gripper (grasp phase)  
    close = dist <= 0.05
    
    open_reward = openness * far.float()  # open when far
    close_reward = (1 - openness) * close.float()  # closed when close
    
    return open_reward * 0.3 + close_reward * 0.7


def finger_contact_reward(env):
    """Dense: reward when object is between the fingers (both sides have proximity).
    
    Uses the object's position relative to EE: if object is at EE height
    and within finger width, it's "between fingers".
    """
    openness, ee_pos, obj_pos = _get_finger_state(env)
    
    # Horizontal distance (XY) between EE and object
    xy_dist = torch.norm(ee_pos[:, :2] - obj_pos[:, :2], dim=1)
    # Vertical alignment (Z)
    z_diff = torch.abs(ee_pos[:, 2] - obj_pos[:, 2])
    
    # Object is "between fingers" if horizontally close AND vertically aligned
    between = (xy_dist < 0.03) & (z_diff < 0.03)
    
    # Additional: reward closing fingers while object is between them
    closing_while_between = between.float() * (1 - openness)
    
    return between.float() * 0.4 + closing_while_between * 0.6


def grasp_success_reward(env):
    """Binary: is the object actually lifted off the table?
    This is the ground truth — everything else is shaping toward this."""
    obj: RigidObject = env.scene["cube_1"]
    # Object default z varies by scene, use env_origins
    height = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    lifted = height > 0.04  # above table surface
    return lifted.float()


def transport_reward(env, std=0.08):
    """Dense: horizontal distance between lifted cube1 and cube2."""
    obj1: RigidObject = env.scene["cube_1"]
    obj2: RigidObject = env.scene["cube_2"]
    height = obj1.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    lifted = height > 0.04
    xy_dist = torch.norm(obj1.data.root_pos_w[:, :2] - obj2.data.root_pos_w[:, :2], dim=1)
    return lifted.float() * (1 - torch.tanh(xy_dist / std))


# Per-env grasp tracking — did this env grasp the cube at any point?
_grasp_tracker = None

def _init_grasp_tracker(num_envs, device):
    global _grasp_tracker
    _grasp_tracker = torch.zeros(num_envs, dtype=torch.bool, device=device)

def _update_grasp_tracker(env):
    """Call every step to track if cube was lifted."""
    global _grasp_tracker
    if _grasp_tracker is None:
        _init_grasp_tracker(env.num_envs, env.device)
    obj: RigidObject = env.scene["cube_1"]
    height = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    currently_lifted = height > 0.08  # well above table + cube2
    _grasp_tracker = _grasp_tracker | currently_lifted

def _reset_grasp_tracker(env_ids):
    global _grasp_tracker
    if _grasp_tracker is not None:
        _grasp_tracker[env_ids] = False


def stack_reward(env):
    """Stacking reward gated on ACTUAL grasping (cube was lifted >8cm at some point)."""
    _update_grasp_tracker(env)
    obj1: RigidObject = env.scene["cube_1"]
    obj2: RigidObject = env.scene["cube_2"]
    xy_dist = torch.norm(obj1.data.root_pos_w[:, :2] - obj2.data.root_pos_w[:, :2], dim=1)
    z_above = obj1.data.root_pos_w[:, 2] - obj2.data.root_pos_w[:, 2]
    stacked = (xy_dist < 0.04) & (z_above > 0.02) & (z_above < 0.10)
    # Only count if cube was grasped (lifted >8cm) at some point this episode
    return (stacked & _grasp_tracker).float()


def action_penalty(env):
    return torch.sum(torch.square(
        env.action_manager.action - env.action_manager.prev_action), dim=1)


# ══════════════════════════════════════════════════════════════════════════════
# Curriculum
# ══════════════════════════════════════════════════════════════════════════════

def grasp_curriculum(env, env_ids):
    if not hasattr(grasp_curriculum, "_n"):
        grasp_curriculum._n = 0
        grasp_curriculum._difficulty = torch.zeros(env.num_envs, device=env.device)
    grasp_curriculum._n += 1

    cfg = _load_cfg()
    d = grasp_curriculum._difficulty

    if isinstance(env_ids, torch.Tensor):
        ids = env_ids
    else:
        ids = torch.tensor(env_ids, device=env.device, dtype=torch.long) if not isinstance(env_ids, slice) else torch.arange(env.num_envs, device=env.device)

    # Update per-env difficulty based on stacking success
    if ids.numel() > 0 and grasp_curriculum._n > 2:
        _update_grasp_tracker(env)
        obj1: RigidObject = env.scene["cube_1"]
        obj2: RigidObject = env.scene["cube_2"]
        xy_dist = torch.norm(obj1.data.root_pos_w[ids, :2] - obj2.data.root_pos_w[ids, :2], dim=1)
        z_above = obj1.data.root_pos_w[ids, 2] - obj2.data.root_pos_w[ids, 2]
        stacked = (xy_dist < 0.04) & (z_above > 0.02) & (z_above < 0.10) & _grasp_tracker[ids]
        # Promote on stack success (make harder = cube2 farther)
        # Demote on failure (make easier = cube2 closer)
        step_up = cfg.get("step_up", 0.02)
        step_down = cfg.get("step_down", 0.003)
        promotion_only = d[ids] < cfg.get("promotion_only_until", 0.3)
        demote = torch.where(promotion_only, torch.zeros_like(d[ids]), torch.full_like(d[ids], step_down))
        delta = torch.where(stacked, step_up, -demote)
        d[ids] = (d[ids] + delta).clamp(0.0, 1.0)

    # Apply reward weights from config
    if grasp_curriculum._n % 50 == 0:
        weights = cfg.get("reward_weights", {})
        for name, w in weights.items():
            try:
                idx = env.reward_manager._term_names.index(name)
                env.reward_manager._term_cfgs[idx].weight = w
            except (ValueError, IndexError):
                pass

    mean_d = d.mean().item()
    d10 = torch.quantile(d, 0.1).item()
    d90 = torch.quantile(d, 0.9).item()

    # Compute metrics
    with torch.no_grad():
        approach_v = approach_reward(env).mean().item()
        grasp_v = grasp_success_reward(env).mean().item()
        stack_v = stack_reward(env).mean().item()

    return {
        "approach": approach_v,
        "grasp_success": grasp_v,
        "stack_success": stack_v,
        "mean_difficulty": mean_d,
        "d10": d10,
        "d90": d90,
        "config_version": cfg.get("version", 0),
    }


def grasp_reset(env, env_ids, spawn_range=0.03):
    """Reset with configurable spawn + strategic cube2 positioning."""
    reset_scene_to_default(env, env_ids)
    _reset_grasp_tracker(env_ids)
    cfg = _load_cfg()
    sr = cfg.get("spawn_range", spawn_range)
    
    cube1: RigidObject = env.scene["cube_1"]
    cube2: RigidObject = env.scene["cube_2"]
    n = len(env_ids)
    
    # Cube1 spawn
    c1 = cube1.data.default_root_state[env_ids].clone()
    for i in range(n):
        c1[i, 0] += torch.empty(1).uniform_(-sr, sr).item()
        c1[i, 1] += torch.empty(1).uniform_(-sr * 2, sr * 2).item()
    c1[:, 0:3] += env.scene.env_origins[env_ids]
    c1[:, 7:13] = 0
    cube1.write_root_pose_to_sim(c1[:, :7], env_ids=env_ids)
    cube1.write_root_velocity_to_sim(c1[:, 7:13], env_ids=env_ids)
    
    # Cube2: per-env distance curriculum
    # d=0: cube2 right next to cube1 (easy stacking)
    # d=1: cube2 at default position (full task)
    c2 = cube2.data.default_root_state[env_ids].clone()
    c2_default = cube2.data.default_root_state[env_ids].clone()
    tracker = getattr(grasp_curriculum, "_difficulty", None)
    if tracker is not None:
        d = tracker[env_ids]
        for i in range(n):
            di = d[i].item()
            max_dist = 0.15
            dist = max_dist * di
            c1_local_x = c1[i, 0].item() - env.scene.env_origins[env_ids[i], 0].item()
            c1_local_y = c1[i, 1].item() - env.scene.env_origins[env_ids[i], 1].item()
            c2_def_x = c2_default[i, 0].item()
            c2_def_y = c2_default[i, 1].item()
            dx = c2_def_x - c1_local_x
            dy = c2_def_y - c1_local_y
            norm = max((dx**2 + dy**2)**0.5, 1e-6)
            c2[i, 0] = c1_local_x + dist * dx / norm + torch.empty(1).uniform_(-0.01*di, 0.01*di).item()
            c2[i, 1] = c1_local_y + dist * dy / norm + torch.empty(1).uniform_(-0.01*di, 0.01*di).item()
    c2[:, 0:3] += env.scene.env_origins[env_ids]
    c2[:, 7:13] = 0
    cube2.write_root_pose_to_sim(c2[:, :7], env_ids=env_ids)
    cube2.write_root_velocity_to_sim(c2[:, 7:13], env_ids=env_ids)


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

@configclass
class GraspRewardsCfg:
    """Grasping-specific rewards for Franka parallel jaw gripper."""
    approach = RewTerm(func=approach_reward, params={"std": 0.08}, weight=1.0)
    pre_grasp = RewTerm(func=pre_grasp_reward, weight=2.0)
    finger_contact = RewTerm(func=finger_contact_reward, weight=3.0)
    grasp_success = RewTerm(func=grasp_success_reward, weight=8.0)
    transport = RewTerm(func=transport_reward, params={"std": 0.08}, weight=0.0)
    stacking = RewTerm(func=stack_reward, weight=0.0)
    action_rate = RewTerm(func=action_penalty, weight=-0.2)

@configclass
class GraspCurriculumCfg:
    grasp = CurrTerm(func=grasp_curriculum)

@configclass
class GraspEventCfg:
    reset = EventTerm(func=grasp_reset, mode="reset", params={"spawn_range": 0.03})
