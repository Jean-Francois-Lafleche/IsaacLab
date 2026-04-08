# LLM-in-the-Loop Adaptive Curriculum for Franka Lift
# Reads parameters from /tmp/adaptive_curriculum_config.json
# The LLM updates this file every 50 iterations based on training analysis

from __future__ import annotations
import json, math, os
from collections.abc import Sequence
from typing import TYPE_CHECKING
import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

CONFIG_PATH = "/tmp/adaptive_curriculum_config.json"
METRICS_PATH = "/tmp/adaptive_metrics.jsonl"


class AdaptivePerEnvDifficulty:
    def __init__(self, num_envs, device):
        self.difficulties = torch.zeros(num_envs, device=device)
        self.device = device
        self._load_config()

    def _load_config(self):
        try:
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)
            self.step_up = cfg.get("step_up", 0.03)
            self.step_down = cfg.get("step_down", 0.01)
            self.promotion_only_until = cfg.get("promotion_only_until", 0.5)
        except:
            self.step_up = 0.03
            self.step_down = 0.01
            self.promotion_only_until = 0.5

    def update(self, env_ids, success):
        if env_ids.numel() == 0:
            return
        # Re-read config periodically (every ~100 calls)
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
    except:
        return {}


def adaptive_curriculum(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> dict:
    if not hasattr(adaptive_curriculum, "_t"):
        adaptive_curriculum._t = AdaptivePerEnvDifficulty(env.num_envs, env.device)
        adaptive_curriculum._n = 0
        adaptive_curriculum._last_log = 0

    t = adaptive_curriculum._t
    adaptive_curriculum._n += 1

    if isinstance(env_ids, torch.Tensor):
        ids = env_ids
    else:
        ids = torch.tensor(env_ids, device=env.device, dtype=torch.long) if not isinstance(env_ids, slice) else torch.arange(env.num_envs, device=env.device)

    if ids.numel() > 0 and adaptive_curriculum._n > 2:
        obj: RigidObject = env.scene["object"]
        lifted = obj.data.root_pos_w[ids, 2] > 0.08
        t.update(ids, lifted)

    d = t.mean_difficulty
    cfg = _load_cfg()

    # Apply configurable parameters
    obj_r_min = cfg.get("obj_range_min", 0.01)
    obj_r_scale = cfg.get("obj_range_scale", 0.09)
    obj_r_power = cfg.get("obj_range_power", 1.0)
    obj_r = obj_r_min + obj_r_scale * (d ** obj_r_power)

    goal_xy_min = cfg.get("goal_xy_min", 0.02)
    goal_xy_scale = cfg.get("goal_xy_scale", 0.23)
    goal_xy_power = cfg.get("goal_xy_power", 1.3)
    goal_offset = goal_xy_min + goal_xy_scale * (d ** goal_xy_power)

    gz_lo = cfg.get("goal_z_lo_start", 0.12) + cfg.get("goal_z_lo_scale", 0.13) * d
    gz_hi = cfg.get("goal_z_hi_start", 0.20) + cfg.get("goal_z_hi_scale", 0.30) * d

    env.event_manager.cfg.reset_object_position.params["pose_range"]["x"] = (-obj_r, obj_r)
    env.event_manager.cfg.reset_object_position.params["pose_range"]["y"] = (-obj_r*2.5, obj_r*2.5)
    cmd = env.command_manager.cfg.object_pose
    cmd.ranges.pos_x = (0.5 - goal_offset, 0.5 + goal_offset)
    cmd.ranges.pos_y = (-goal_offset, goal_offset)
    cmd.ranges.pos_z = (gz_lo, gz_hi)

    # Log metrics to file for LLM analysis (every ~32 curriculum calls ≈ 1 iter)
    d10 = torch.quantile(t.difficulties, 0.1).item()
    d50 = torch.quantile(t.difficulties, 0.5).item()
    d90 = torch.quantile(t.difficulties, 0.9).item()

    return {
        "difficulty": d, "d10": d10, "d50": d50, "d90": d90,
        "obj_range": obj_r, "goal_offset": goal_offset, "goal_z_max": gz_hi,
    }


@configclass
class AdaptiveCurriculumCfg:
    adaptive = CurrTerm(func=adaptive_curriculum)

@configclass
class AdaptiveEventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {"x": (-0.01, 0.01), "y": (-0.025, 0.025), "z": (0.0, 0.0)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object", body_names="Object")})
