# Stack task: Detailed static curriculum with per-env difficulty and spawn range scheduling
# Mirrors the lift task's detailed curriculum pattern, adapted for stacking.

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_rl_env_cfg import FrankaStackRLEnvCfg
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ═══════════════════════════════════════════════════════════════════════
# Per-env difficulty tracker
# ═══════════════════════════════════════════════════════════════════════

class PerEnvDifficulty:
    def __init__(self, num_envs, device, step_up=0.02, step_down=0.005,
                 promotion_only_until=0.5):
        self.difficulties = torch.zeros(num_envs, device=device)
        self.step_up = step_up
        self.step_down = step_down
        self.promotion_only_until = promotion_only_until
        self.device = device

    def update(self, env_ids, success):
        if env_ids.numel() == 0:
            return
        d = self.difficulties[env_ids]
        no_demote = d < self.promotion_only_until
        step_down = torch.where(no_demote, torch.zeros_like(d),
                                torch.full_like(d, self.step_down))
        delta = torch.where(success, self.step_up, -step_down)
        self.difficulties[env_ids] = (d + delta).clamp(0.0, 1.0)

    @property
    def mean_difficulty(self):
        return self.difficulties.mean().item()


# ═══════════════════════════════════════════════════════════════════════
# Parameter schedules: spawn range and separation
# ═══════════════════════════════════════════════════════════════════════

def _spawn_x_range(d):
    """X spawn range: (0.45,0.55) easy → (0.4,0.6) full"""
    lo = 0.45 - 0.05 * d
    hi = 0.55 + 0.05 * d
    return (lo, hi)

def _spawn_y_range(d):
    """Y spawn range: ±0.02 easy → ±0.10 full"""
    half = 0.02 + 0.08 * d
    return (-half, half)

def _min_separation(d):
    """Min cube separation: 0.06 easy → 0.10 full"""
    return 0.06 + 0.04 * d


def detailed_stack_curriculum(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> dict:
    """Progressive difficulty curriculum for stacking.
    
    Monitors whether cube1 is lifted (intermediate success signal) and
    whether stacking is achieved (full success).
    Gradually widens cube spawn range and separation.
    """
    if not hasattr(detailed_stack_curriculum, "_state"):
        detailed_stack_curriculum._state = {
            "tracker": PerEnvDifficulty(env.num_envs, env.device, step_up=0.02, step_down=0.005),
            "n": 0,
        }

    state = detailed_stack_curriculum._state
    t = state["tracker"]
    state["n"] += 1

    if isinstance(env_ids, torch.Tensor):
        ids = env_ids
    elif isinstance(env_ids, slice):
        ids = torch.arange(env.num_envs, device=env.device)
    else:
        ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)

    # Update difficulty based on lifting success (intermediate signal)
    if ids.numel() > 0 and state["n"] > 2:
        cube_1: RigidObject = env.scene["cube_1"]
        cube1_z = cube_1.data.root_pos_w[ids, 2] - env.scene.env_origins[ids, 2]
        lifted = cube1_z > 0.04
        t.update(ids, lifted)

    d = t.mean_difficulty

    # Apply spawn range scheduling
    x_range = _spawn_x_range(d)
    y_range = _spawn_y_range(d)
    min_sep = _min_separation(d)

    # Update the event manager's cube randomization parameters
    evt = env.event_manager.cfg.randomize_cube_positions
    evt.params["pose_range"]["x"] = x_range
    evt.params["pose_range"]["y"] = y_range
    evt.params["min_separation"] = min_sep

    return {
        "difficulty": d,
        "spawn_x": x_range,
        "spawn_y": y_range,
        "min_sep": min_sep,
    }


# ═══════════════════════════════════════════════════════════════════════
# Config classes
# ═══════════════════════════════════════════════════════════════════════

@configclass
class DetailedStackCurriculumCfg:
    detailed = CurrTerm(func=detailed_stack_curriculum)


@configclass
class DetailedStackEventCfg:
    """Events starting with tight spawn range."""
    init_franka_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.0400, 0.0400],
        },
    )
    randomize_franka_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    randomize_cube_positions = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.45, 0.55), "y": (-0.02, 0.02), "z": (0.0203, 0.0203), "yaw": (-1.0, 1.0)},
            "min_separation": 0.06,
            "asset_cfgs": [SceneEntityCfg("cube_1"), SceneEntityCfg("cube_2"), SceneEntityCfg("cube_3")],
        },
    )


@configclass
class FrankaStackRLDetailedEnvCfg(FrankaStackRLEnvCfg):
    """Franka cube stack with detailed static curriculum."""

    def __post_init__(self):
        super().__post_init__()
        self.events = DetailedStackEventCfg()
        self.curriculum = DetailedStackCurriculumCfg()
