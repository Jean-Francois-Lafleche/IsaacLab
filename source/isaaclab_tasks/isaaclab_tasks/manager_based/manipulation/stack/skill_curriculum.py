# Skill-Decomposed Curriculum for Franka Stack
#
# APPROACH: Train all skills jointly in ONE policy. The curriculum controls:
# 1. Which reward terms are active (skill-specific)
# 2. What initial states are used (output of previous skill)
# 3. When to progress to the next skill (mastery threshold)
#
# SKILL DEPENDENCY: REACH → GRASP → LIFT → TRANSPORT → PLACE
#
# Each skill has: reward, success metric, initial state setup, curriculum
# Skills are cumulative — once activated, they stay active forever.
# The agent learns to chain skills naturally because later skills
# require earlier skills to set up their starting conditions.

from __future__ import annotations
import math
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


# ══════════════════════════════════════════════════════════════════════════════
# Reward functions — one per skill
# ══════════════════════════════════════════════════════════════════════════════

def reach_reward(env: ManagerBasedRLEnv, std: float = 0.1,
                 ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
                 obj_cfg: SceneEntityCfg = SceneEntityCfg("cube_1")) -> torch.Tensor:
    """Distance between end-effector and cube1."""
    ee: FrameTransformer = env.scene[ee_cfg.name]
    obj: RigidObject = env.scene[obj_cfg.name]
    dist = torch.norm(ee.data.target_pos_w[..., 0, :] - obj.data.root_pos_w, dim=1)
    return 1 - torch.tanh(dist / std)


def grasp_reward(env: ManagerBasedRLEnv,
                 obj_cfg: SceneEntityCfg = SceneEntityCfg("cube_1")) -> torch.Tensor:
    """Binary: is cube1 lifted even slightly? (proxy for grasping)"""
    obj: RigidObject = env.scene[obj_cfg.name]
    # Cube default z is ~0.055. If it's above 0.065, something is holding it up.
    grasped = (obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]) > 0.065
    return grasped.float()


def lift_reward(env: ManagerBasedRLEnv, target_height: float = 0.15,
                std: float = 0.1,
                obj_cfg: SceneEntityCfg = SceneEntityCfg("cube_1")) -> torch.Tensor:
    """Reward for lifting cube1 to target height."""
    obj: RigidObject = env.scene[obj_cfg.name]
    height = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    # Only reward if actually grasped (above table)
    grasped = height > 0.065
    height_err = torch.abs(height - target_height)
    return grasped.float() * (1 - torch.tanh(height_err / std))


def transport_reward(env: ManagerBasedRLEnv, std: float = 0.1,
                     obj1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
                     obj2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2")) -> torch.Tensor:
    """Reward for moving cube1 horizontally above cube2 (while lifted)."""
    obj1: RigidObject = env.scene[obj1_cfg.name]
    obj2: RigidObject = env.scene[obj2_cfg.name]
    # Only reward if cube1 is lifted
    height = obj1.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    lifted = height > 0.08
    # Horizontal distance between cubes
    xy_dist = torch.norm(obj1.data.root_pos_w[:, :2] - obj2.data.root_pos_w[:, :2], dim=1)
    return lifted.float() * (1 - torch.tanh(xy_dist / std))


def place_reward(env: ManagerBasedRLEnv,
                 obj1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
                 obj2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2")) -> torch.Tensor:
    """Binary: is cube1 stacked on cube2?"""
    obj1: RigidObject = env.scene[obj1_cfg.name]
    obj2: RigidObject = env.scene[obj2_cfg.name]
    xy_dist = torch.norm(obj1.data.root_pos_w[:, :2] - obj2.data.root_pos_w[:, :2], dim=1)
    z_above = obj1.data.root_pos_w[:, 2] - obj2.data.root_pos_w[:, 2]
    stacked = (xy_dist < 0.04) & (z_above > 0.03) & (z_above < 0.12)
    return stacked.float()


def action_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


# ══════════════════════════════════════════════════════════════════════════════
# Skill-phase tracker
# ══════════════════════════════════════════════════════════════════════════════

SKILL_NAMES = ["reach", "grasp", "lift", "transport", "place"]
SKILL_THRESHOLDS = {
    "reach": 0.6,      # reach reward > 0.6 → activate grasp
    "grasp": 0.3,      # grasp reward > 0.3 → activate lift
    "lift": 0.3,       # lift reward > 0.3 → activate transport
    "transport": 0.3,  # transport reward > 0.3 → activate place
    "place": None,      # terminal skill
}
SKILL_MIN_ITERS = 30  # min iters before checking promotion


class SkillPhaseTracker:
    """Tracks which skills are active. Skills activate cumulatively —
    once a skill is learned, it stays active and the next one activates."""
    
    def __init__(self, device):
        self.device = device
        self.active_skills = {"reach": True, "grasp": False, "lift": False, 
                              "transport": False, "place": False}
        self.skill_iters = {s: 0 for s in SKILL_NAMES}
        self.skill_ema = {s: 0.0 for s in SKILL_NAMES}  # EMA of each skill's reward
        self.ema_alpha = 0.05
        self.call_count = 0
    
    def get_active_phase(self) -> str:
        """Return the highest active skill (the current learning focus)."""
        for s in reversed(SKILL_NAMES):
            if self.active_skills[s]:
                return s
        return "reach"
    
    def update(self, rewards: dict[str, float]):
        """Check if we should activate the next skill."""
        self.call_count += 1
        
        for skill in SKILL_NAMES:
            if skill in rewards:
                self.skill_ema[skill] = (
                    (1 - self.ema_alpha) * self.skill_ema[skill] + 
                    self.ema_alpha * rewards[skill]
                )
            
            if self.active_skills[skill]:
                self.skill_iters[skill] += 1
        
        # Check promotion: activate next skill if current is mastered
        for i, skill in enumerate(SKILL_NAMES[:-1]):
            next_skill = SKILL_NAMES[i + 1]
            if (self.active_skills[skill] 
                and not self.active_skills[next_skill]
                and self.skill_iters[skill] >= SKILL_MIN_ITERS
                and self.skill_ema[skill] >= SKILL_THRESHOLDS[skill]):
                self.active_skills[next_skill] = True


# ══════════════════════════════════════════════════════════════════════════════
# Curriculum function
# ══════════════════════════════════════════════════════════════════════════════

def skill_curriculum(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> dict:
    if not hasattr(skill_curriculum, "_tracker"):
        skill_curriculum._tracker = SkillPhaseTracker(env.device)
    
    tracker = skill_curriculum._tracker
    
    if isinstance(env_ids, torch.Tensor):
        ids = env_ids
    else:
        ids = torch.tensor(env_ids, device=env.device, dtype=torch.long) if not isinstance(env_ids, slice) else torch.arange(env.num_envs, device=env.device)
    
    # Compute current skill rewards for promotion check
    if ids.numel() > 0:
        rewards = {}
        with torch.no_grad():
            rewards["reach"] = reach_reward(env).mean().item()
            rewards["grasp"] = grasp_reward(env).mean().item()
            rewards["lift"] = lift_reward(env).mean().item()
            rewards["transport"] = transport_reward(env).mean().item()
            rewards["place"] = place_reward(env).mean().item()
        tracker.update(rewards)
    
    # Dynamically adjust reward weights based on active skills
    # Active skills get their full weight; inactive skills get 0
    weights = {
        "reaching": 1.0 if tracker.active_skills["reach"] else 0.0,
        "grasping": 5.0 if tracker.active_skills["grasp"] else 0.0,
        "lifting": 8.0 if tracker.active_skills["lift"] else 0.0,
        "transporting": 12.0 if tracker.active_skills["transport"] else 0.0,
        "placing": 15.0 if tracker.active_skills["place"] else 0.0,
    }
    
    # Apply weights via _term_cfgs (safe — only modifying weight attribute)
    for term_name, weight in weights.items():
        try:
            idx = env.reward_manager._term_names.index(term_name)
            env.reward_manager._term_cfgs[idx].weight = weight
        except (ValueError, IndexError):
            pass
    
    phase = tracker.get_active_phase()
    phase_idx = SKILL_NAMES.index(phase)
    
    return {
        "phase": float(phase_idx),
        "reach_ema": tracker.skill_ema["reach"],
        "grasp_ema": tracker.skill_ema["grasp"],
        "lift_ema": tracker.skill_ema["lift"],
        "transport_ema": tracker.skill_ema["transport"],
        "place_ema": tracker.skill_ema["place"],
        "num_active_skills": sum(1 for v in tracker.active_skills.values() if v),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Reset function — initial state depends on active skills
# ══════════════════════════════════════════════════════════════════════════════

def skill_reset(env, env_ids, dummy=0):
    """Reset with initial conditions appropriate for the current skill phase.
    
    When learning REACH: cube near robot (easy to reach)
    When learning GRASP: cube very close to gripper
    When learning LIFT+: cube in default position (agent must reach + grasp first)
    """
    tracker = getattr(skill_curriculum, "_tracker", None)
    
    # Default reset first
    reset_scene_to_default(env, env_ids)
    
    if tracker is None:
        return
    
    cube1: RigidObject = env.scene["cube_1"]
    robot: Articulation = env.scene["robot"]
    n = len(env_ids)
    
    phase = tracker.get_active_phase()
    
    # Set cube1 spawn based on phase
    cube1_states = cube1.data.default_root_state[env_ids].clone()
    
    if phase == "reach":
        # Cube close to robot, small randomization
        for i in range(n):
            cube1_states[i, 0] += torch.empty(1).uniform_(-0.03, 0.03).item()
            cube1_states[i, 1] += torch.empty(1).uniform_(-0.05, 0.05).item()
    elif phase == "grasp":
        # Cube very close to default (will be near gripper after reaching)
        for i in range(n):
            cube1_states[i, 0] += torch.empty(1).uniform_(-0.05, 0.05).item()
            cube1_states[i, 1] += torch.empty(1).uniform_(-0.08, 0.08).item()
    else:
        # Later phases: wider spawn (agent has reach+grasp skills)
        spread = min(1.0, (SKILL_NAMES.index(phase) - 1) / 3.0)
        for i in range(n):
            cube1_states[i, 0] += torch.empty(1).uniform_(-0.1 * spread, 0.1 * spread).item()
            cube1_states[i, 1] += torch.empty(1).uniform_(-0.25 * spread, 0.25 * spread).item()
    
    cube1_states[:, 0:3] += env.scene.env_origins[env_ids]
    cube1_states[:, 7:13] = 0
    cube1.write_root_pose_to_sim(cube1_states[:, :7], env_ids=env_ids)
    cube1.write_root_velocity_to_sim(cube1_states[:, 7:13], env_ids=env_ids)


# ══════════════════════════════════════════════════════════════════════════════
# Config classes  
# ══════════════════════════════════════════════════════════════════════════════

@configclass
class SkillRewardsCfg:
    """All skill rewards — weights controlled dynamically by curriculum."""
    reaching = RewTerm(func=reach_reward, params={"std": 0.1}, weight=1.0)
    grasping = RewTerm(func=grasp_reward, weight=0.0)  # starts OFF
    lifting = RewTerm(func=lift_reward, params={"target_height": 0.15, "std": 0.1}, weight=0.0)
    transporting = RewTerm(func=transport_reward, params={"std": 0.1}, weight=0.0)
    placing = RewTerm(func=place_reward, weight=0.0)  # starts OFF
    action_rate = RewTerm(func=action_penalty, weight=-0.3)

@configclass
class SkillCurriculumCfg:
    skills = CurrTerm(func=skill_curriculum)

@configclass
class SkillEventCfg:
    reset_all = EventTerm(func=skill_reset, mode="reset",
        params={"dummy": 0})
