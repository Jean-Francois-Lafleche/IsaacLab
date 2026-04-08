# LLM-Crafted Curriculum v4 for Dexsuite Kuka-Allegro Lift
#
# v4 = v3 (per-env smooth + failure sampling) + "sticky retry"
#
# New in v4:
#   On failure, the env resets to its EXACT same initial conditions.
#   The agent must retry the same scenario until it succeeds (or max_retries hit).
#   This forces the policy to solve every configuration it encounters.
#
# Inspired by Prioritized Level Replay (PLR), but simpler:
#   PLR: maintains a buffer of high-regret levels, samples from it
#   Sticky retry: literally keeps the env at its failed state until solved
#
# Overfitting mitigation: max_retries=5 before resampling fresh

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import combine_frame_transforms

from isaaclab_tasks.manager_based.manipulation.dexsuite import mdp
from isaaclab_tasks.manager_based.manipulation.dexsuite.dexsuite_env_cfg import (
    EventCfg as DexsuiteEventCfg,
)

from isaaclab_tasks.manager_based.manipulation.dexsuite.llm_curriculum_v2 import (
    _gravity_schedule,
    _obj_spawn_x_schedule,
    _obj_spawn_y_schedule,
    _obj_spawn_z_schedule,
    _obj_rotation_schedule,
    _goal_x_schedule,
    _goal_y_schedule,
    _goal_z_schedule,
    _joint_reset_range_schedule,
    PerEnvDifficulty,
)

from isaaclab_tasks.manager_based.manipulation.dexsuite.llm_curriculum_v3 import (
    SpatialFailureTracker,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ──────────────────────────────────────────────────────────────────────────────
# Sticky retry tracker — stores initial states and retry counts
# ──────────────────────────────────────────────────────────────────────────────
class StickyRetryTracker:
    """Tracks initial states and retry counts for the sticky retry mechanism.
    
    On failure: env resets to the SAME initial conditions (up to max_retries).
    On success OR max retries exceeded: env gets fresh conditions.
    """

    def __init__(self, num_envs: int, device: torch.device, max_retries: int = 5):
        self.num_envs = num_envs
        self.device = device
        self.max_retries = max_retries

        # Per-env stored initial object poses (7: x,y,z,qx,qy,qz,qw)
        self.stored_obj_poses = torch.zeros(num_envs, 7, device=device)
        # Per-env stored joint positions
        self.stored_joint_pos = None  # initialized on first use (unknown joint count)
        # Retry count per env
        self.retry_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        # Whether each env has a stored state to retry
        self.has_stored_state = torch.zeros(num_envs, dtype=torch.bool, device=device)
        # Track last episode's success for each env
        self.last_success = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def should_retry(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Returns boolean mask: True = retry same state, False = sample fresh."""
        if env_ids.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)

        failed = ~self.last_success[env_ids]
        has_state = self.has_stored_state[env_ids]
        under_limit = self.retry_count[env_ids] < self.max_retries

        return failed & has_state & under_limit

    def record_outcome(self, env_ids: torch.Tensor, success: torch.Tensor):
        """Record whether each env succeeded."""
        self.last_success[env_ids] = success

    def store_state(self, env_ids: torch.Tensor, obj_pose: torch.Tensor,
                    joint_pos: torch.Tensor):
        """Store initial conditions for potential retry."""
        self.stored_obj_poses[env_ids] = obj_pose
        if self.stored_joint_pos is None:
            self.stored_joint_pos = torch.zeros(
                self.num_envs, joint_pos.shape[-1], device=self.device)
        self.stored_joint_pos[env_ids] = joint_pos
        self.has_stored_state[env_ids] = True

    def mark_retry(self, env_ids: torch.Tensor):
        """Increment retry counter for envs that are retrying."""
        self.retry_count[env_ids] += 1

    def mark_fresh(self, env_ids: torch.Tensor):
        """Reset retry counter for envs getting fresh conditions."""
        self.retry_count[env_ids] = 0
        self.has_stored_state[env_ids] = False


# ──────────────────────────────────────────────────────────────────────────────
# Sticky reset — retries failed states, samples fresh on success
# ──────────────────────────────────────────────────────────────────────────────
def sticky_reset_object(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Reset object with sticky retry: failed envs get same state, succeeded get fresh."""
    asset: RigidObject = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()
    n = len(env_ids)

    state = getattr(llm_curriculum_v4, "_state", None)

    if state is None or state.get("warmup", True):
        # Uniform fallback during warmup
        range_list = [pose_range.get(key, (0.0, 0.0))
                      for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (n, 6), device=asset.device)
        positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
        orientations_delta = math_utils.quat_from_euler_xyz(
            rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    else:
        tracker = state["difficulty"]
        retry = state["retry"]
        d = tracker.difficulties[env_ids]

        # Determine which envs retry vs get fresh
        retry_mask = retry.should_retry(env_ids)
        fresh_mask = ~retry_mask
        retry_ids = env_ids[retry_mask]
        fresh_ids = env_ids[fresh_mask]
        fresh_local = torch.where(fresh_mask)[0]  # local indices into env_ids

        # Initialize output
        positions = root_states[:, 0:3] + env.scene.env_origins[env_ids]
        orientations = root_states[:, 3:7].clone()

        # RETRY envs: restore stored poses
        if retry_ids.numel() > 0:
            retry_local = torch.where(retry_mask)[0]
            stored = retry.stored_obj_poses[retry_ids]
            positions[retry_local] = stored[:, :3]
            orientations[retry_local] = stored[:, 3:]
            retry.mark_retry(retry_ids)

        # FRESH envs: sample new poses based on difficulty
        if fresh_ids.numel() > 0:
            n_fresh = fresh_ids.numel()
            d_fresh = d[fresh_mask]
            rand_samples = torch.zeros(n_fresh, 6, device=asset.device)

            for i in range(n_fresh):
                di = d_fresh[i].item()
                x_lo, x_hi = _obj_spawn_x_schedule(di)
                y_lo, y_hi = _obj_spawn_y_schedule(di)
                z_lo, z_hi = _obj_spawn_z_schedule(di)
                rot_range = _obj_rotation_schedule(di)
                rand_samples[i, 0] = torch.empty(1, device=asset.device).uniform_(x_lo, x_hi)
                rand_samples[i, 1] = torch.empty(1, device=asset.device).uniform_(y_lo, y_hi)
                rand_samples[i, 2] = torch.empty(1, device=asset.device).uniform_(z_lo, z_hi)
                rand_samples[i, 3] = torch.empty(1, device=asset.device).uniform_(-rot_range, rot_range)
                rand_samples[i, 4] = torch.empty(1, device=asset.device).uniform_(-rot_range, rot_range)
                rand_samples[i, 5] = torch.empty(1, device=asset.device).uniform_(-rot_range, rot_range)

            fresh_root = asset.data.default_root_state[fresh_ids].clone()
            fresh_pos = fresh_root[:, 0:3] + env.scene.env_origins[fresh_ids] + rand_samples[:, 0:3]
            fresh_orn_delta = math_utils.quat_from_euler_xyz(
                rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
            fresh_orn = math_utils.quat_mul(fresh_root[:, 3:7], fresh_orn_delta)

            positions[fresh_local] = fresh_pos
            orientations[fresh_local] = fresh_orn
            retry.mark_fresh(fresh_ids)

        # Store states for potential future retry
        stored_poses = torch.cat([positions, orientations], dim=-1)
        retry.store_state(env_ids, stored_poses,
                          env.scene["robot"].data.default_joint_pos[env_ids])

    # Velocities: always zero
    vel_range_list = [velocity_range.get(key, (0.0, 0.0))
                      for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    vel_ranges = torch.tensor(vel_range_list, device=asset.device)
    vel_samples = math_utils.sample_uniform(
        vel_ranges[:, 0], vel_ranges[:, 1], (n, 6), device=asset.device)
    velocities = root_states[:, 7:13] + vel_samples

    asset.write_root_pose_to_sim(
        torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def sticky_reset_joints(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    position_range: list[float],
    velocity_range: list[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset joints with sticky retry — failed envs get same joint positions."""
    asset: Articulation = env.scene[asset_cfg.name]
    state = getattr(llm_curriculum_v4, "_state", None)

    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()

    if state is not None and not state.get("warmup", True):
        retry = state["retry"]
        tracker = state["difficulty"]
        retry_mask = retry.should_retry(env_ids)

        # RETRY envs: restore stored joints
        if retry_mask.any() and retry.stored_joint_pos is not None:
            retry_local = torch.where(retry_mask)[0]
            retry_ids = env_ids[retry_mask]
            joint_pos[retry_local] = retry.stored_joint_pos[retry_ids]

        # FRESH envs: sample based on difficulty
        fresh_mask = ~retry_mask
        if fresh_mask.any():
            fresh_local = torch.where(fresh_mask)[0]
            d = tracker.difficulties[env_ids[fresh_mask]]
            for i in range(fresh_local.numel()):
                r = _joint_reset_range_schedule(d[i].item())
                joint_pos[fresh_local[i]] += torch.empty_like(
                    joint_pos[fresh_local[i]]).uniform_(-r, r)
    else:
        noise = math_utils.sample_uniform(
            position_range[0], position_range[1], joint_pos.shape, joint_pos.device)
        joint_pos += noise

    joint_pos = joint_pos.clamp_(
        asset.data.joint_pos_limits[env_ids, :, 0],
        asset.data.joint_pos_limits[env_ids, :, 1])
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


# ──────────────────────────────────────────────────────────────────────────────
# Main curriculum function
# ──────────────────────────────────────────────────────────────────────────────
def llm_curriculum_v4(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
) -> dict[str, float]:
    """LLM Curriculum v4: v3 + sticky retry on failure."""
    if not hasattr(llm_curriculum_v4, "_state"):
        llm_curriculum_v4._state = {
            "difficulty": PerEnvDifficulty(
                num_envs=env.num_envs, device=env.device,
                step_up=0.04, step_down=0.01,
                promotion_only_until=0.5,
            ),
            "failure": SpatialFailureTracker(
                device=env.device, num_envs=env.num_envs,
                x_bins=4, y_bins=4, z_bins=4, ema_alpha=0.05,
            ),
            "retry": StickyRetryTracker(
                num_envs=env.num_envs, device=env.device,
                max_retries=5,
            ),
            "warmup": True,
            "warmup_calls": 0,
            "call_count": 0,
        }

    state = llm_curriculum_v4._state
    tracker = state["difficulty"]
    failure = state["failure"]
    retry = state["retry"]
    state["call_count"] += 1

    if isinstance(env_ids, torch.Tensor):
        reset_ids = env_ids
    elif isinstance(env_ids, slice):
        reset_ids = torch.arange(env.num_envs, device=env.device)
    else:
        reset_ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)

    # Record outcomes
    if reset_ids.numel() > 0 and not state["warmup"]:
        robot: Articulation = env.scene["robot"]
        obj: RigidObject = env.scene["object"]
        command = env.command_manager.get_command("object_pose")
        des_pos_w, _ = combine_frame_transforms(
            robot.data.root_pos_w[reset_ids], robot.data.root_quat_w[reset_ids],
            command[reset_ids, :3], command[reset_ids, 3:7]
        )
        pos_dist = torch.norm(des_pos_w - obj.data.root_pos_w[reset_ids], dim=1)
        success = pos_dist < 0.15
        tracker.update(reset_ids, success)
        failure.record_outcomes(reset_ids, success)
        retry.record_outcome(reset_ids, success)

    if state["warmup"]:
        state["warmup_calls"] += 1
        if state["warmup_calls"] >= 3:
            state["warmup"] = False

    # Apply gravity
    mean_d = tracker.mean_difficulty
    gz = _gravity_schedule(mean_d)
    env.event_manager.cfg.variable_gravity.params["gravity_distribution_params"] = (
        (0.0, 0.0, gz), (0.0, 0.0, gz)
    )

    # Apply goal range
    gx = _goal_x_schedule(mean_d)
    gy = _goal_y_schedule(mean_d)
    gz_goal = _goal_z_schedule(mean_d)
    cmd_cfg = env.command_manager.cfg.object_pose
    cmd_cfg.ranges.pos_x = gx
    cmd_cfg.ranges.pos_y = gy
    cmd_cfg.ranges.pos_z = gz_goal

    # Stats
    d = tracker.difficulties
    retry_rate = (retry.retry_count > 0).float().mean().item()
    avg_retries = retry.retry_count.float().mean().item()

    return {
        "mean_difficulty": mean_d,
        "d10": torch.quantile(d, 0.1).item(),
        "d50_median": torch.quantile(d, 0.5).item(),
        "d90": torch.quantile(d, 0.9).item(),
        "gravity_z": gz,
        "goal_z_max": gz_goal[1],
        "retry_rate": retry_rate,
        "avg_retries": avg_retries,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Config classes
# ──────────────────────────────────────────────────────────────────────────────
@configclass
class LLMv4CurriculumCfg:
    """LLM Curriculum v4 — per-env smooth + sticky retry on failure."""
    llm_v4 = CurrTerm(func=llm_curriculum_v4)


@configclass
class LLMv4EventCfg(DexsuiteEventCfg):
    """Events with sticky retry resets."""

    reset_object = EventTerm(
        func=sticky_reset_object,
        mode="reset",
        params={
            "pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (0.0, 0.0),
                           "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    variable_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="reset",
        params={
            "gravity_distribution_params": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            "operation": "abs",
        },
    )

    reset_robot_joints = EventTerm(
        func=sticky_reset_joints,
        mode="reset",
        params={
            "position_range": [-0.1, 0.1],
            "velocity_range": [0.0, 0.0],
        },
    )
