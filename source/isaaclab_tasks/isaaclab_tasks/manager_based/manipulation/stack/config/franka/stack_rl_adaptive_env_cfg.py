# Stack task: Adaptive LLM-in-the-loop curriculum
# Reads configurable parameters from /tmp/stack_adaptive_config.json
# The LLM can update this file during training to adjust difficulty.

from __future__ import annotations

import json
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

CONFIG_PATH = "/tmp/stack_adaptive_config.json"


class AdaptivePerEnvDifficulty:
    def __init__(self, num_envs, device):
        self.difficulties = torch.zeros(num_envs, device=device)
        self.device = device
        self._load_config()

    def _load_config(self):
        try:
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)
            self.step_up = cfg.get("step_up", 0.02)
            self.step_down = cfg.get("step_down", 0.005)
            self.promotion_only_until = cfg.get("promotion_only_until", 0.5)
        except Exception:
            self.step_up = 0.02
            self.step_down = 0.005
            self.promotion_only_until = 0.5

    def update(self, env_ids, success):
        if env_ids.numel() == 0:
            return
        if not hasattr(self, '_update_count'):
            self._update_count = 0
        self._update_count += 1
        if self._update_count % 100 == 0:
            self._load_config()

        d = self.difficulties[env_ids]
        no_demote = d < self.promotion_only_until
        step_down = torch.where(no_demote, torch.zeros_like(d),
                                torch.full_like(d, self.step_down))
        delta = torch.where(success, self.step_up, -step_down)
        self.difficulties[env_ids] = (d + delta).clamp(0.0, 1.0)

    @property
    def mean_difficulty(self):
        return self.difficulties.mean().item()


def _load_cfg():
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def adaptive_stack_curriculum(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> dict:
    """Adaptive curriculum that reads spawn/difficulty params from a JSON config file."""
    if not hasattr(adaptive_stack_curriculum, "_t"):
        adaptive_stack_curriculum._t = AdaptivePerEnvDifficulty(env.num_envs, env.device)
        adaptive_stack_curriculum._n = 0

    t = adaptive_stack_curriculum._t
    adaptive_stack_curriculum._n += 1

    if isinstance(env_ids, torch.Tensor):
        ids = env_ids
    elif isinstance(env_ids, slice):
        ids = torch.arange(env.num_envs, device=env.device)
    else:
        ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)

    # Update difficulty based on lifting success
    if ids.numel() > 0 and adaptive_stack_curriculum._n > 2:
        cube_1: RigidObject = env.scene["cube_1"]
        cube1_z = cube_1.data.root_pos_w[ids, 2] - env.scene.env_origins[ids, 2]
        lifted = cube1_z > 0.04
        t.update(ids, lifted)

    d = t.mean_difficulty
    cfg = _load_cfg()

    # Configurable spawn scheduling
    x_lo_start = cfg.get("spawn_x_lo_start", 0.45)
    x_lo_scale = cfg.get("spawn_x_lo_scale", -0.05)
    x_hi_start = cfg.get("spawn_x_hi_start", 0.55)
    x_hi_scale = cfg.get("spawn_x_hi_scale", 0.05)
    x_range = (x_lo_start + x_lo_scale * d, x_hi_start + x_hi_scale * d)

    y_half_start = cfg.get("spawn_y_half_start", 0.02)
    y_half_scale = cfg.get("spawn_y_half_scale", 0.08)
    y_power = cfg.get("spawn_y_power", 1.0)
    y_half = y_half_start + y_half_scale * (d ** y_power)
    y_range = (-y_half, y_half)

    min_sep_start = cfg.get("min_sep_start", 0.06)
    min_sep_scale = cfg.get("min_sep_scale", 0.04)
    min_sep = min_sep_start + min_sep_scale * d

    # Apply spawn scheduling
    evt = env.event_manager.cfg.randomize_cube_positions
    evt.params["pose_range"]["x"] = x_range
    evt.params["pose_range"]["y"] = y_range
    evt.params["min_separation"] = min_sep

    # Difficulty quantiles
    d10 = torch.quantile(t.difficulties, 0.1).item()
    d50 = torch.quantile(t.difficulties, 0.5).item()
    d90 = torch.quantile(t.difficulties, 0.9).item()

    return {
        "difficulty": d,
        "d10": d10,
        "d50": d50,
        "d90": d90,
        "spawn_x": x_range,
        "spawn_y": y_range,
        "min_sep": min_sep,
    }


# ═══════════════════════════════════════════════════════════════════════
# Config classes
# ═══════════════════════════════════════════════════════════════════════

@configclass
class AdaptiveStackCurriculumCfg:
    adaptive = CurrTerm(func=adaptive_stack_curriculum)


@configclass
class AdaptiveStackEventCfg:
    """Events starting with tight spawn range (LLM will adjust)."""
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
class FrankaStackRLAdaptiveEnvCfg(FrankaStackRLEnvCfg):
    """Franka cube stack with LLM-adaptive curriculum."""

    def __post_init__(self):
        super().__post_init__()
        self.events = AdaptiveStackEventCfg()
        self.curriculum = AdaptiveStackCurriculumCfg()
