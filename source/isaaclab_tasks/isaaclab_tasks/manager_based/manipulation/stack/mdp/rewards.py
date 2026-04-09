# Copyright (c) 2024, Stack RL Experiment
# Reward functions for the cube stacking task with Franka robot.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reaching_cube1(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
) -> torch.Tensor:
    """Reward for reaching cube_1 using tanh kernel on ee-to-cube1 distance."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube_1: RigidObject = env.scene[cube_1_cfg.name]

    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    cube1_pos = cube_1.data.root_pos_w

    distance = torch.norm(cube1_pos - ee_pos, dim=1)
    return 1.0 - torch.tanh(distance / std)


def grasping_cube1(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    diff_threshold: float = 0.06,
) -> torch.Tensor:
    """Binary reward: is cube1 grasped (close to ee and gripper closed)?"""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube_1: RigidObject = env.scene[cube_1_cfg.name]

    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    cube1_pos = cube_1.data.root_pos_w
    pose_diff = torch.linalg.vector_norm(cube1_pos - ee_pos, dim=1)

    gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    grasped = torch.logical_and(
        pose_diff < diff_threshold,
        torch.abs(
            robot.data.joint_pos[:, gripper_joint_ids[0]]
            - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32, device=env.device)
        )
        > env.cfg.gripper_threshold,
    )
    grasped = torch.logical_and(
        grasped,
        torch.abs(
            robot.data.joint_pos[:, gripper_joint_ids[1]]
            - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32, device=env.device)
        )
        > env.cfg.gripper_threshold,
    )

    return grasped.float()


def lifting_cube1(
    env: ManagerBasedRLEnv,
    minimal_height: float = 0.04,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
) -> torch.Tensor:
    """Binary reward: is cube1 lifted above minimal_height (relative to table)?"""
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    # cube init z is ~0.0203 on table; heights are world-frame relative to env origin
    cube1_z = cube_1.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return torch.where(cube1_z > minimal_height, 1.0, 0.0)


def cube1_above_cube2(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    minimal_height: float = 0.04,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
) -> torch.Tensor:
    """Tanh reward on horizontal distance between cube1 and cube2, only when cube1 is lifted."""
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]

    cube1_pos = cube_1.data.root_pos_w
    cube2_pos = cube_2.data.root_pos_w

    # Only reward when cube1 is lifted
    cube1_z = cube1_pos[:, 2] - env.scene.env_origins[:, 2]
    is_lifted = cube1_z > minimal_height

    # Horizontal (XY) distance between cube1 and cube2
    xy_dist = torch.norm(cube1_pos[:, :2] - cube2_pos[:, :2], dim=1)
    proximity = 1.0 - torch.tanh(xy_dist / std)

    return is_lifted.float() * proximity


def stacking_success(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    xy_threshold: float = 0.05,
    height_threshold: float = 0.005,
    height_diff: float = 0.0468,
) -> torch.Tensor:
    """Binary reward: is cube1 stacked on cube2 (and gripper open)?"""
    robot: Articulation = env.scene[robot_cfg.name]
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]

    pos_diff = cube_1.data.root_pos_w - cube_2.data.root_pos_w
    xy_dist = torch.norm(pos_diff[:, :2], dim=1)
    h_dist = torch.abs(pos_diff[:, 2])

    # Cube1 should be above cube2 with correct height difference
    stacked = torch.logical_and(xy_dist < xy_threshold, torch.abs(h_dist - height_diff) < height_threshold)
    stacked = torch.logical_and(pos_diff[:, 2] > 0.0, stacked)  # cube1 above cube2

    # Gripper should be open (released)
    gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    stacked = torch.logical_and(
        torch.isclose(
            robot.data.joint_pos[:, gripper_joint_ids[0]],
            torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32, device=env.device),
            atol=1e-3, rtol=1e-3,
        ),
        stacked,
    )
    stacked = torch.logical_and(
        torch.isclose(
            robot.data.joint_pos[:, gripper_joint_ids[1]],
            torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32, device=env.device),
            atol=1e-3, rtol=1e-3,
        ),
        stacked,
    )

    return stacked.float()


def action_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large action rate (L2 norm of action difference)."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def joint_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize large joint velocities (L2)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel), dim=1)
