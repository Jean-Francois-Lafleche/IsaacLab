"""Diagnostic reward: prints actual scene heights at iter 0."""
from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import FrameTransformer
from isaaclab.managers import RewardTermCfg as RewTerm, CurriculumTermCfg as CurrTerm, EventTermCfg as EventTerm, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp import reset_scene_to_default

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_printed = [False]

def diag_reward(env) -> torch.Tensor:
    ee: FrameTransformer = env.scene["ee_frame"]
    cube1: RigidObject = env.scene["cube_1"]
    cube2: RigidObject = env.scene["cube_2"]
    robot: Articulation = env.scene["robot"]
    origins = env.scene.env_origins
    
    ee_pos_w = ee.data.target_pos_w  # (N, num_targets, 3)
    c1_pos_w = cube1.data.root_pos_w
    c2_pos_w = cube2.data.root_pos_w
    c1_vel = cube1.data.root_lin_vel_w
    
    ee_pos = ee_pos_w[:, 0, :]  # EE center
    
    # Heights relative to env_origins
    c1_h = c1_pos_w[:, 2] - origins[:, 2]
    c2_h = c2_pos_w[:, 2] - origins[:, 2]
    ee_h = ee_pos[:, 2] - origins[:, 2]
    
    # EE to cube1 distance
    ee_to_c1 = torch.norm(c1_pos_w - ee_pos, dim=1)
    
    # Finger positions
    if ee_pos_w.shape[1] >= 3:
        rf_pos = ee_pos_w[:, 1, :]  # right finger
        lf_pos = ee_pos_w[:, 2, :]  # left finger
        rf_h = rf_pos[:, 2] - origins[:, 2]
        lf_h = lf_pos[:, 2] - origins[:, 2]
    
    # Gripper
    f1_idx = robot.data.joint_names.index("panda_finger_joint1")
    f2_idx = robot.data.joint_names.index("panda_finger_joint2")
    finger_pos = robot.data.joint_pos[:, [f1_idx, f2_idx]]
    
    if not _printed[0]:
        _printed[0] = True
        i = 0  # Sample env 0
        print(f"\n{'='*60}")
        print(f"DIAGNOSTIC at iter 0, env 0:")
        print(f"  env_origin: {origins[i].tolist()}")
        print(f"  cube1 world pos: {c1_pos_w[i].tolist()}")
        print(f"  cube2 world pos: {c2_pos_w[i].tolist()}")
        print(f"  cube1 height (rel to origin): {c1_h[i].item():.6f}")
        print(f"  cube2 height (rel to origin): {c2_h[i].item():.6f}")
        print(f"  ee center world pos: {ee_pos[i].tolist()}")
        print(f"  ee height (rel to origin): {ee_h[i].item():.6f}")
        print(f"  ee to cube1 dist: {ee_to_c1[i].item():.6f}")
        if ee_pos_w.shape[1] >= 3:
            print(f"  right finger pos: {rf_pos[i].tolist()}")
            print(f"  left finger pos: {lf_pos[i].tolist()}")
            print(f"  right finger h: {rf_h[i].item():.6f}")
            print(f"  left finger h: {lf_h[i].item():.6f}")
        print(f"  finger joints: {finger_pos[i].tolist()}")
        print(f"  cube1 velocity: {c1_vel[i].tolist()}")
        print(f"  MEANS over all envs:")
        print(f"    c1_h: {c1_h.mean().item():.6f} ± {c1_h.std().item():.6f}")
        print(f"    c2_h: {c2_h.mean().item():.6f} ± {c2_h.std().item():.6f}")
        print(f"    ee_h: {ee_h.mean().item():.6f}")
        print(f"    ee_to_c1: {ee_to_c1.mean().item():.6f}")
        print(f"    finger_open: {finger_pos.mean(dim=1).mean().item():.6f}")
        print(f"{'='*60}\n")
    
    # Simple reach reward so training doesn't crash
    return 1.0 - torch.tanh(ee_to_c1 / 0.1)

def diag_curriculum(env, env_ids):
    return {}

def diag_reset(env, env_ids):
    reset_scene_to_default(env, env_ids)

@configclass
class DiagRewardsCfg:
    reach = RewTerm(func=diag_reward, weight=1.0)

@configclass
class DiagCurriculumCfg:
    diag = CurrTerm(func=diag_curriculum)

@configclass
class DiagEventCfg:
    reset = EventTerm(func=diag_reset, mode="reset")
