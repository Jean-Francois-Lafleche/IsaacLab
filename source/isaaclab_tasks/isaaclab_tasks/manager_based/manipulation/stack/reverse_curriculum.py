# Reverse Curriculum for Franka Stack
#
# PRINCIPLE: Start from success, work backwards.
#
# Stage 5: Cube grasped, 1cm above target → just release
# Stage 4: Cube grasped, 5cm above target → lower + release
# Stage 3: Cube grasped, 15cm above → transport down + release
# Stage 2: Cube grasped, on table → lift + transport + release
# Stage 1: Cube in open gripper (touching) → close + lift + transport + release
# Stage 0: Cube on table, arm at default → reach + grasp + lift + transport + release
#
# Uses per-env difficulty (0→1). d=1.0 = stage 5 (easiest), d=0.0 = stage 0 (hardest).
# REVERSED: we start at d=1.0 and work DOWN to d=0.0.

from __future__ import annotations
import json, math
from collections.abc import Sequence
from typing import TYPE_CHECKING
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CurriculumTermCfg as CurrTerm, EventTermCfg as EventTerm, SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp import reset_scene_to_default
from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Cube 2 default position (the target — bottom cube stays fixed)
CUBE2_POS = (0.5, 0.0, 0.055)  # on the table
# "Success" position: cube 1 on top of cube 2
STACK_HEIGHT = 0.055 + 0.05 + 0.005  # cube2_z + cube_size + small gap

# Franka joint positions for "gripper above cube2, holding cube1"
# These are approximate — the robot's EE will be roughly above cube2
GRASP_JOINTS = [0.0, -0.3, 0.0, -2.2, 0.0, 2.5, 0.741]  # arm joints
GRIPPER_CLOSED = [0.01, 0.01]  # fingers closed on cube
GRIPPER_OPEN = [0.04, 0.04]  # fingers open


class ReversePerEnvDifficulty:
    """Tracks per-env difficulty. d=1.0 is EASIEST (near success), d=0.0 is HARDEST (full task).
    
    Difficulty DECREASES on success (make it harder) and INCREASES on failure (make it easier).
    This is the REVERSE of the standard curriculum.
    """
    def __init__(self, num_envs, device, step_down=0.02, step_up=0.01):
        # Start at d=1.0 (easiest)
        self.difficulties = torch.ones(num_envs, device=device)
        self.step_down = step_down  # decrease on success (make harder)
        self.step_up = step_up      # increase on failure (make easier)
        self.device = device

    def update(self, env_ids, success):
        if env_ids.numel() == 0:
            return
        d = self.difficulties[env_ids]
        # SUCCESS → decrease difficulty (make harder, closer to full task)
        # FAILURE → increase difficulty (make easier, closer to success start)
        delta = torch.where(success, -self.step_down, self.step_up)
        self.difficulties[env_ids] = (d + delta).clamp(0.0, 1.0)

    @property
    def mean_difficulty(self):
        return self.difficulties.mean().item()


def reverse_reset_scene(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    position_range: dict,
    velocity_range: dict,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset based on reverse curriculum difficulty.
    
    d=1.0: Cube1 already stacked on cube2, gripper open nearby → learn "don't knock off"
    d=0.7: Cube1 on table RIGHT NEXT to cube2 → learn to nudge/place on top
    d=0.4: Cube1 on table near robot → learn to pick up + carry + place
    d=0.15: Cube1 on table, moderate distance → reach + pick + carry + place
    d=0.0: Full task — cube1 at random position, arm at default
    """
    state = getattr(reverse_curriculum, "_state", None)
    if state is None:
        state = getattr(reverse_curriculum_adaptive, "_state", None)
    robot: Articulation = env.scene["robot"]
    cube1: RigidObject = env.scene["cube_1"]
    cube2: RigidObject = env.scene["cube_2"]
    n = len(env_ids)

    # Always reset cube2 to default position
    cube2_states = cube2.data.default_root_state[env_ids].clone()
    cube2_states[:, 0:3] += env.scene.env_origins[env_ids]
    cube2_states[:, 7:13] = 0
    cube2.write_root_pose_to_sim(cube2_states[:, :7], env_ids=env_ids)
    cube2.write_root_velocity_to_sim(cube2_states[:, 7:13], env_ids=env_ids)

    # Reset robot root pose + joints to default
    robot_states = robot.data.default_root_state[env_ids].clone()
    robot_states[:, 0:3] += env.scene.env_origins[env_ids]
    robot.write_root_pose_to_sim(robot_states[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(robot_states[:, 7:13], env_ids=env_ids)

    default_joint_pos = robot.data.default_joint_pos[env_ids].clone()
    default_joint_vel = robot.data.default_joint_vel[env_ids].clone()
    default_joint_vel.zero_()

    # Cube1 + robot positioning based on difficulty
    cube1_states = cube1.data.default_root_state[env_ids].clone()
    cube1_states[:, 7:13] = 0  # zero velocity

    if state is None:
        # No curriculum — use default positions
        cube1_states[:, 0:3] += env.scene.env_origins[env_ids]
        cube1.write_root_pose_to_sim(cube1_states[:, :7], env_ids=env_ids)
        cube1.write_root_velocity_to_sim(cube1_states[:, 7:13], env_ids=env_ids)
        default_joint_pos += math_utils.sample_uniform(-0.125, 0.125, default_joint_pos.shape, default_joint_pos.device)
        default_joint_pos = default_joint_pos.clamp_(robot.data.joint_pos_limits[env_ids, :, 0], robot.data.joint_pos_limits[env_ids, :, 1])
        robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)
        return

    tracker = state["difficulty"]
    d = tracker.difficulties[env_ids]
    cube2_world = cube2_states[:, 0:3]  # cube2 positions in world frame

    for i in range(n):
        di = d[i].item()
        c2x = cube2_world[i, 0].item()
        c2y = cube2_world[i, 1].item()
        c2z = cube2_world[i, 2].item()

        if di > 0.7:
            # EASIEST: Cube1 already stacked on cube2 (resting on top)
            jitter = (1.0 - di) / 0.3 * 0.03  # 0 to 3cm xy jitter
            cube1_states[i, 0] = c2x + torch.empty(1).uniform_(-jitter, jitter).item()
            cube1_states[i, 1] = c2y + torch.empty(1).uniform_(-jitter, jitter).item()
            cube1_states[i, 2] = c2z + 0.055  # resting on top

        elif di > 0.4:
            # MEDIUM: Cube1 on table near cube2
            dist = (0.7 - di) / 0.3 * 0.10 + 0.02  # 2cm to 12cm away
            angle = torch.empty(1).uniform_(0, 6.28).item()
            cube1_states[i, 0] = c2x + dist * math.cos(angle)
            cube1_states[i, 1] = c2y + dist * math.sin(angle)
            cube1_states[i, 2] = c2z  # on table

        elif di > 0.15:
            # HARDER: Cube1 farther on table
            dist = (0.4 - di) / 0.25 * 0.15 + 0.05  # 5cm to 20cm away
            angle = torch.empty(1).uniform_(0, 6.28).item()
            cube1_states[i, 0] = c2x + dist * math.cos(angle)
            cube1_states[i, 1] = c2y + dist * math.sin(angle)
            cube1_states[i, 2] = c2z

        else:
            # HARDEST: Full randomization (default range)
            spread = (0.15 - di) / 0.15
            cube1_states[i, 0] = c2x + torch.empty(1).uniform_(-0.1 * spread, 0.1 * spread).item()
            cube1_states[i, 1] = c2y + torch.empty(1).uniform_(-0.25 * spread, 0.25 * spread).item()
            cube1_states[i, 2] = c2z

        # Add small joint noise proportional to task difficulty
        noise_scale = 0.05 + (1.0 - di) * 0.2  # more noise at harder difficulty
        default_joint_pos[i] += torch.empty_like(default_joint_pos[i]).uniform_(-noise_scale, noise_scale)

    # Write states
    default_joint_pos = default_joint_pos.clamp_(robot.data.joint_pos_limits[env_ids, :, 0], robot.data.joint_pos_limits[env_ids, :, 1])
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)
    cube1.write_root_pose_to_sim(cube1_states[:, :7], env_ids=env_ids)
    cube1.write_root_velocity_to_sim(cube1_states[:, 7:13], env_ids=env_ids)


def reverse_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
) -> dict[str, float]:
    """Reverse curriculum: start easy (near success), work backwards to full task."""
    if not hasattr(reverse_curriculum, "_state"):
        reverse_curriculum._state = {
            "difficulty": ReversePerEnvDifficulty(
                env.num_envs, env.device,
                step_down=0.03,  # success → make harder (faster push)
                step_up=0.003,   # failure → barely ease up
            ),
            "call_count": 0,
        }

    state = reverse_curriculum._state
    tracker = state["difficulty"]
    state["call_count"] += 1

    if isinstance(env_ids, torch.Tensor):
        ids = env_ids
    elif isinstance(env_ids, slice):
        ids = torch.arange(env.num_envs, device=env.device)
    else:
        ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)

    # Check stacking success for resetting envs
    if ids.numel() > 0 and state["call_count"] > 2:
        cube1: RigidObject = env.scene["cube_1"]
        cube2: RigidObject = env.scene["cube_2"]
        c1_pos = cube1.data.root_pos_w[ids]
        c2_pos = cube2.data.root_pos_w[ids]
        # Stacked: cube1 is above cube2 within tolerance
        xy_dist = torch.norm(c1_pos[:, :2] - c2_pos[:, :2], dim=1)
        z_above = c1_pos[:, 2] - c2_pos[:, 2]
        stacked = (xy_dist < 0.05) & (z_above > 0.03) & (z_above < 0.15)
        tracker.update(ids, stacked)

    d = tracker.mean_difficulty
    d_all = tracker.difficulties
    
    return {
        "mean_difficulty": d,
        "d10": torch.quantile(d_all, 0.1).item(),
        "d50": torch.quantile(d_all, 0.5).item(),
        "d90": torch.quantile(d_all, 0.9).item(),
        "stage": 1.0 - d,  # 0=easiest, 1=hardest for intuitive reading
    }


# Adaptive version that reads config from file
ADAPTIVE_CONFIG = "/tmp/stack_adaptive_config.json"

def reverse_curriculum_adaptive(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
) -> dict[str, float]:
    """Same as reverse_curriculum but reads step sizes from config file."""
    if not hasattr(reverse_curriculum_adaptive, "_state"):
        try:
            with open(ADAPTIVE_CONFIG) as f:
                cfg = json.load(f)
            sd = cfg.get("step_down", 0.02)
            su = cfg.get("step_up", 0.005)
        except:
            sd, su = 0.02, 0.005
        
        reverse_curriculum_adaptive._state = {
            "difficulty": ReversePerEnvDifficulty(
                env.num_envs, env.device, step_down=sd, step_up=su,
            ),
            "call_count": 0,
        }

    state = reverse_curriculum_adaptive._state
    tracker = state["difficulty"]
    state["call_count"] += 1

    # Re-read config periodically
    if state["call_count"] % 200 == 0:
        try:
            with open(ADAPTIVE_CONFIG) as f:
                cfg = json.load(f)
            tracker.step_down = cfg.get("step_down", 0.02)
            tracker.step_up = cfg.get("step_up", 0.005)
        except:
            pass

    if isinstance(env_ids, torch.Tensor):
        ids = env_ids
    elif isinstance(env_ids, slice):
        ids = torch.arange(env.num_envs, device=env.device)
    else:
        ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)

    if ids.numel() > 0 and state["call_count"] > 2:
        cube1: RigidObject = env.scene["cube_1"]
        cube2: RigidObject = env.scene["cube_2"]
        c1_pos = cube1.data.root_pos_w[ids]
        c2_pos = cube2.data.root_pos_w[ids]
        xy_dist = torch.norm(c1_pos[:, :2] - c2_pos[:, :2], dim=1)
        z_above = c1_pos[:, 2] - c2_pos[:, 2]
        stacked = (xy_dist < 0.05) & (z_above > 0.03) & (z_above < 0.15)
        tracker.update(ids, stacked)

    d = tracker.mean_difficulty
    d_all = tracker.difficulties
    
    return {
        "mean_difficulty": d,
        "d10": torch.quantile(d_all, 0.1).item(),
        "d50": torch.quantile(d_all, 0.5).item(),
        "d90": torch.quantile(d_all, 0.9).item(),
        "stage": 1.0 - d,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Config classes
# ──────────────────────────────────────────────────────────────────────────────
@configclass
class ReverseStaticCurriculumCfg:
    reverse = CurrTerm(func=reverse_curriculum)

@configclass
class ReverseAdaptiveCurriculumCfg:
    reverse_adaptive = CurrTerm(func=reverse_curriculum_adaptive)

@configclass
class ReverseEventCfg:
    """Events that use reverse curriculum to set initial state.
    No reset_scene_to_default — we handle everything in reverse_reset_scene."""
    reverse_reset = EventTerm(
        func=reverse_reset_scene,
        mode="reset",
        params={
            "position_range": {},
            "velocity_range": {},
        },
    )
