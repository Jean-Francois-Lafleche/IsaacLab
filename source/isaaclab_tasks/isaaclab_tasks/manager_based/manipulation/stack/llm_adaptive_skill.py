# LLM-in-the-Loop Adaptive Skill Training for Franka Stack
#
# The LLM writes to /tmp/llm_trainer_config.json every 100 iterations.
# This curriculum reads that file and applies the settings dynamically.
# All reward weights, spawn ranges, and skill thresholds are configurable.

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
# Reward functions — each reads its weight from config dynamically
# ══════════════════════════════════════════════════════════════════════════════

def reach_reward(env, std=0.1, 
                 ee_cfg=SceneEntityCfg("ee_frame"),
                 obj_cfg=SceneEntityCfg("cube_1")):
    ee: FrameTransformer = env.scene[ee_cfg.name]
    obj: RigidObject = env.scene[obj_cfg.name]
    dist = torch.norm(ee.data.target_pos_w[..., 0, :] - obj.data.root_pos_w, dim=1)
    cfg = _load_cfg()
    std = cfg.get("reach_std", std)
    return 1 - torch.tanh(dist / std)


def grasp_reward(env, obj_cfg=SceneEntityCfg("cube_1"),
                 ee_cfg=SceneEntityCfg("ee_frame")):
    """Dense grasp reward: combines finger proximity + lift detection."""
    obj: RigidObject = env.scene[obj_cfg.name]
    ee: FrameTransformer = env.scene[ee_cfg.name]
    # Component 1: ee-to-object closeness (finger proximity)
    dist = torch.norm(ee.data.target_pos_w[..., 0, :] - obj.data.root_pos_w, dim=1)
    close = (dist < 0.05).float()  # very close to object
    # Component 2: object lifted (grasped successfully)
    height = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    lifted = (height > 0.065).float()
    # Dense reward: proximity when close + big bonus when actually lifted
    cfg = _load_cfg()
    close_weight = cfg.get("grasp_close_weight", 0.3)
    lift_weight = cfg.get("grasp_lift_weight", 0.7)
    return close * close_weight + lifted * lift_weight


def lift_reward(env, target_height=0.15, std=0.05,
                obj_cfg=SceneEntityCfg("cube_1")):
    obj: RigidObject = env.scene[obj_cfg.name]
    height = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    grasped = height > 0.065
    cfg = _load_cfg()
    target_height = cfg.get("lift_target_height", target_height)
    height_err = torch.abs(height - target_height)
    return grasped.float() * (1 - torch.tanh(height_err / std))


def transport_reward(env, std=0.08,
                     obj1_cfg=SceneEntityCfg("cube_1"),
                     obj2_cfg=SceneEntityCfg("cube_2")):
    obj1: RigidObject = env.scene[obj1_cfg.name]
    obj2: RigidObject = env.scene[obj2_cfg.name]
    height = obj1.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    lifted = height > 0.08
    xy_dist = torch.norm(obj1.data.root_pos_w[:, :2] - obj2.data.root_pos_w[:, :2], dim=1)
    return lifted.float() * (1 - torch.tanh(xy_dist / std))


def place_reward(env, obj1_cfg=SceneEntityCfg("cube_1"),
                 obj2_cfg=SceneEntityCfg("cube_2")):
    obj1: RigidObject = env.scene[obj1_cfg.name]
    obj2: RigidObject = env.scene[obj2_cfg.name]
    xy_dist = torch.norm(obj1.data.root_pos_w[:, :2] - obj2.data.root_pos_w[:, :2], dim=1)
    z_above = obj1.data.root_pos_w[:, 2] - obj2.data.root_pos_w[:, 2]
    stacked = (xy_dist < 0.04) & (z_above > 0.03) & (z_above < 0.12)
    return stacked.float()


def action_penalty(env):
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


# ══════════════════════════════════════════════════════════════════════════════
# Curriculum — reads all config from file, applies dynamically
# ══════════════════════════════════════════════════════════════════════════════

def llm_adaptive_curriculum(env, env_ids):
    if not hasattr(llm_adaptive_curriculum, "_n"):
        llm_adaptive_curriculum._n = 0
    llm_adaptive_curriculum._n += 1
    
    cfg = _load_cfg()
    
    if isinstance(env_ids, torch.Tensor):
        ids = env_ids
    else:
        ids = torch.tensor(env_ids, device=env.device, dtype=torch.long) if not isinstance(env_ids, slice) else torch.arange(env.num_envs, device=env.device)
    
    # Apply reward weights from config (every 50 curriculum calls ≈ every ~2 iters)
    if llm_adaptive_curriculum._n % 50 == 0:
        weights = cfg.get("reward_weights", {})
        for term_name, weight in weights.items():
            try:
                idx = env.reward_manager._term_names.index(term_name)
                env.reward_manager._term_cfgs[idx].weight = weight
            except (ValueError, IndexError):
                pass
    
    # Apply spawn range from config
    spawn_range = cfg.get("spawn_range", 0.05)
    env.event_manager.cfg.skill_reset.params["spawn_range"] = spawn_range
    
    # Compute metrics for logging
    with torch.no_grad():
        reach_val = reach_reward(env).mean().item()
        grasp_val = grasp_reward(env).mean().item()
        lift_val = lift_reward(env).mean().item()
        transport_val = transport_reward(env).mean().item()
        place_val = place_reward(env).mean().item()
    
    return {
        "reach_reward": reach_val,
        "grasp_reward": grasp_val,
        "lift_reward": lift_val,
        "transport_reward": transport_val,
        "place_reward": place_val,
        "config_version": cfg.get("version", 0),
    }


def skill_adaptive_reset(env, env_ids, spawn_range=0.05):
    """Reset with configurable spawn range."""
    reset_scene_to_default(env, env_ids)
    
    cfg = _load_cfg()
    spawn_range = cfg.get("spawn_range", spawn_range)
    
    cube1: RigidObject = env.scene["cube_1"]
    n = len(env_ids)
    
    cube1_states = cube1.data.default_root_state[env_ids].clone()
    for i in range(n):
        cube1_states[i, 0] += torch.empty(1).uniform_(-spawn_range, spawn_range).item()
        cube1_states[i, 1] += torch.empty(1).uniform_(-spawn_range * 2.5, spawn_range * 2.5).item()
    
    cube1_states[:, 0:3] += env.scene.env_origins[env_ids]
    cube1_states[:, 7:13] = 0
    cube1.write_root_pose_to_sim(cube1_states[:, :7], env_ids=env_ids)
    cube1.write_root_velocity_to_sim(cube1_states[:, 7:13], env_ids=env_ids)


# ══════════════════════════════════════════════════════════════════════════════
# Config classes
# ══════════════════════════════════════════════════════════════════════════════

@configclass
class LLMAdaptiveRewardsCfg:
    reaching = RewTerm(func=reach_reward, params={"std": 0.1}, weight=1.0)
    grasping = RewTerm(func=grasp_reward, weight=3.0)
    lifting = RewTerm(func=lift_reward, params={"target_height": 0.15, "std": 0.05}, weight=0.0)
    transporting = RewTerm(func=transport_reward, params={"std": 0.08}, weight=0.0)
    placing = RewTerm(func=place_reward, weight=0.0)
    action_rate = RewTerm(func=action_penalty, weight=-0.3)

@configclass
class LLMAdaptiveCurriculumCfg:
    adaptive = CurrTerm(func=llm_adaptive_curriculum)

@configclass
class LLMAdaptiveEventCfg:
    skill_reset = EventTerm(func=skill_adaptive_reset, mode="reset",
        params={"spawn_range": 0.03})
