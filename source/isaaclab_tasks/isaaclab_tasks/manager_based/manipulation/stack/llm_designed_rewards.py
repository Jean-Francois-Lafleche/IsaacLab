# LLM-Designed Rewards for Franka Cube Stacking
# ================================================
# ALL rewards designed from first principles by understanding:
#   - Franka Panda: 7-DOF arm + parallel-jaw gripper (panda_finger_*)
#   - ee_frame: FrameTransformer with end-effector at panda_hand + 10.34cm offset
#   - cube_1 (blue, to manipulate): init (0.4, 0.0, 0.0203), height ~0.046m
#   - cube_2 (red, target): init (0.55, 0.05, 0.0203)
#   - Gripper: open_val=0.04, close=0.0, threshold=0.005
#   - Table at z=0 (env_origins), cube bottom at z=0.0203
#   - Cube top at ~0.043m (half-height = 0.023)
#
# NO imports from mdp/rewards.py — every reward is written here.

from __future__ import annotations
import json
from typing import TYPE_CHECKING
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import FrameTransformer
from isaaclab.managers import CurriculumTermCfg as CurrTerm, EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp import reset_scene_to_default

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

CONFIG_PATH = "/tmp/llm_trainer_config.json"

def _load_cfg():
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# Per-env episode state tracking
# ══════════════════════════════════════════════════════════════════════════════

_episode_state = {}

def _get_state(env):
    """Lazy-init per-env episode state."""
    if "was_lifted" not in _episode_state:
        dev = env.device
        n = env.num_envs
        _episode_state["was_lifted"] = torch.zeros(n, dtype=torch.bool, device=dev)
        _episode_state["stable_steps"] = torch.zeros(n, dtype=torch.long, device=dev)
        _episode_state["max_height"] = torch.zeros(n, dtype=torch.float32, device=dev)
    return _episode_state

def _reset_state(env_ids):
    if "was_lifted" in _episode_state:
        _episode_state["was_lifted"][env_ids] = False
        _episode_state["stable_steps"][env_ids] = 0
        _episode_state["max_height"][env_ids] = 0.0

def _update_state(env):
    s = _get_state(env)
    cube1: RigidObject = env.scene["cube_1"]
    h = cube1.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    # "lifted" = cube1 is clearly above its resting position
    # Resting: z ≈ 0.020. Lifted threshold: 0.06 (well above table + cube half-height)
    # Anti-gaming: cube starts at 0.020, so 0.06 is unreachable at t=0 ✓
    s["was_lifted"] = s["was_lifted"] | (h > 0.06)
    s["max_height"] = torch.max(s["max_height"], h)


# ══════════════════════════════════════════════════════════════════════════════
# Reward 1: REACH — Move end-effector toward cube1
# ══════════════════════════════════════════════════════════════════════════════

def llm_reach(env, std: float = 0.1,
              ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
              cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_1")) -> torch.Tensor:
    """Dense reaching reward using tanh kernel on ee→cube1 distance.
    
    Design rationale:
      - End-effector position from FrameTransformer (target_pos_w[:, 0, :])
      - Cube position from RigidObject root_pos_w
      - tanh kernel: smooth, bounded [0,1], peaks at 0 distance
      - std=0.1 gives good gradient at typical reaching distances (~10-20cm)
    
    Validation:
      t=0: ee starts ~15cm from cube → reward ≈ 0.3 (provides gradient) ✓
      success: ee at cube → reward ≈ 1.0 ✓
      gaming: no way to game — distance is Euclidean ✓
    """
    ee: FrameTransformer = env.scene[ee_cfg.name]
    cube: RigidObject = env.scene[cube_cfg.name]
    ee_pos = ee.data.target_pos_w[:, 0, :]  # end-effector world position
    cube_pos = cube.data.root_pos_w          # cube center world position
    dist = torch.norm(cube_pos - ee_pos, dim=1)
    return 1.0 - torch.tanh(dist / std)


# ══════════════════════════════════════════════════════════════════════════════
# Reward 2: GRASP — Achieve a firm grip on cube1
# ══════════════════════════════════════════════════════════════════════════════

def llm_grasp(env, proximity_threshold: float = 0.06,
              robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
              ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
              cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_1")) -> torch.Tensor:
    """Binary grasp detection: ee close to cube AND fingers closed.
    
    Design rationale:
      - Franka gripper: panda_finger_* joints, open=0.04, closed=0.0
      - "Grasping" = fingers not at open position (deviated by > threshold)
      - BOTH fingers must be closed (AND condition prevents single-finger gaming)
      - ee must be close to cube (prevents "close gripper in air" gaming)
    
    Validation:
      t=0: gripper open (0.04), so |0.04 - 0.04| = 0 < 0.005 → NOT grasping → reward = 0 ✓
      success: gripper closed (0.0), |0.0 - 0.04| = 0.04 > 0.005, ee near cube → reward = 1.0 ✓
      gaming: close gripper far from cube → proximity check fails → reward = 0 ✓
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee: FrameTransformer = env.scene[ee_cfg.name]
    cube: RigidObject = env.scene[cube_cfg.name]

    # Check ee-to-cube proximity
    ee_pos = ee.data.target_pos_w[:, 0, :]
    cube_pos = cube.data.root_pos_w
    dist = torch.norm(cube_pos - ee_pos, dim=1)
    is_close = dist < proximity_threshold

    # Check gripper closure (both fingers must deviate from open position)
    gripper_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    open_val = torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32, device=env.device)
    threshold = env.cfg.gripper_threshold

    finger_0_closed = torch.abs(robot.data.joint_pos[:, gripper_ids[0]] - open_val) > threshold
    finger_1_closed = torch.abs(robot.data.joint_pos[:, gripper_ids[1]] - open_val) > threshold

    grasped = is_close & finger_0_closed & finger_1_closed
    return grasped.float()


# ══════════════════════════════════════════════════════════════════════════════
# Reward 3: LIFT — Raise cube1 well above the table
# ══════════════════════════════════════════════════════════════════════════════

def llm_lift(env, min_height: float = 0.06,
             cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_1")) -> torch.Tensor:
    """Dense lift reward: height above threshold, scaled by how high.
    
    Design rationale:
      - Cube resting z = 0.0203 (center of mass above table)
      - min_height = 0.06 — clearly above resting position (0.0203 + safety margin)
      - Uses tanh on height beyond threshold for smooth gradient upward
      - Gated: returns 0 below min_height (no reward for cube sitting on table)
    
    Validation:
      t=0: cube at z=0.0203 < 0.06 → reward = 0 ✓
      success: cube at z=0.07 → tanh((0.07-0.06)/0.05) ≈ 0.20, good gradient ✓
      gaming: can't lift without grasping (gravity pulls it back) ✓
    """
    _update_state(env)
    cube: RigidObject = env.scene[cube_cfg.name]
    h = cube.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    above = (h - min_height).clamp(min=0)
    return torch.tanh(above / 0.05)  # normalize by 5cm


# ══════════════════════════════════════════════════════════════════════════════
# Reward 4: TRANSPORT — Move lifted cube1 toward cube2 horizontally
# ══════════════════════════════════════════════════════════════════════════════

def llm_transport(env, std: float = 0.08, min_height: float = 0.06,
                  cube1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
                  cube2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2")) -> torch.Tensor:
    """Dense transport reward: xy proximity to cube2, gated on being lifted.
    
    Design rationale:
      - Only reward horizontal approach when cube1 is already lifted
      - tanh kernel on XY distance (ignore Z — that's the lift/place reward)
      - std=0.08 gives good gradient at typical cube separations (~10-15cm)
      - Gating on lift prevents "push cube along table" gaming
    
    Validation:
      t=0: cube1 not lifted → gate = 0 → reward = 0 ✓
      success: cube1 lifted and above cube2 → xy_dist ≈ 0 → reward ≈ 1.0 ✓
      gaming: push cube1 to cube2 on table → not lifted → gate = 0 ✓
    """
    cube1: RigidObject = env.scene[cube1_cfg.name]
    cube2: RigidObject = env.scene[cube2_cfg.name]

    c1_h = cube1.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    is_lifted = (c1_h > min_height).float()

    xy_dist = torch.norm(
        cube1.data.root_pos_w[:, :2] - cube2.data.root_pos_w[:, :2], dim=1
    )
    proximity = 1.0 - torch.tanh(xy_dist / std)

    return is_lifted * proximity


# ══════════════════════════════════════════════════════════════════════════════
# Reward 5: RELEASE — Open gripper when cube1 is positioned above cube2
# ══════════════════════════════════════════════════════════════════════════════

def llm_release(env, xy_thresh: float = 0.06, min_height: float = 0.06,
                robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                cube1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
                cube2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2")) -> torch.Tensor:
    """Dense release reward: incentivize gripper opening when above target.
    
    Design rationale:
      - Gated on: (1) cube1 above cube2 in XY, (2) cube1 lifted, (3) was_lifted this episode
      - was_lifted gate prevents t=0 gaming (gripper starts open at t=0!)
      - Uses squared openness: partial open → small reward, full open → max
        This prevents "62% open equilibrium" where robot half-opens
      - Openness = finger_pos / open_val, clamped to [0,1]
    
    Validation:
      t=0: was_lifted=False → gate = 0 → reward = 0, even though gripper is open ✓
      success: lifted, above cube2, gripper fully open → openness²≈1.0 → full reward ✓
      gaming: open gripper far from cube2 → xy gate fails → reward = 0 ✓
      gaming: never lift, just open gripper → was_lifted gate fails → reward = 0 ✓
    """
    _update_state(env)
    s = _get_state(env)

    robot: Articulation = env.scene[robot_cfg.name]
    cube1: RigidObject = env.scene[cube1_cfg.name]
    cube2: RigidObject = env.scene[cube2_cfg.name]

    c1_pos = cube1.data.root_pos_w
    c2_pos = cube2.data.root_pos_w
    c1_h = c1_pos[:, 2] - env.scene.env_origins[:, 2]

    # Gate: lifted AND horizontally above cube2 AND was_lifted this episode
    xy_close = torch.norm(c1_pos[:, :2] - c2_pos[:, :2], dim=1) < xy_thresh
    is_above = (c1_h > min_height) & xy_close & s["was_lifted"]

    # Gripper openness: mean of both fingers, normalized to [0,1]
    gripper_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    finger_pos = robot.data.joint_pos[:, gripper_ids]
    openness = (finger_pos.mean(dim=1) / env.cfg.gripper_open_val).clamp(0.0, 1.0)

    # Squared: incentivize FULL opening, not partial
    return is_above.float() * openness ** 2


# ══════════════════════════════════════════════════════════════════════════════
# Reward 6: STACK SUCCESS — Full goal state check
# ══════════════════════════════════════════════════════════════════════════════

def llm_stack_success(env, xy_thresh: float = 0.05,
                      height_diff: float = 0.0468, height_tol: float = 0.02,
                      robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                      cube1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
                      cube2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2")) -> torch.Tensor:
    """Binary stacking success: full goal state conjunction.
    
    Design rationale — success requires ALL of:
      1. XY alignment: cube1 horizontally within 5cm of cube2
      2. Height: cube1 above cube2 by one cube height (0.0468m ± 0.02m)
      3. Above: cube1.z > cube2.z (not below)
      4. Gripper open: both fingers near open position
      5. Was manipulated: cube1 was lifted during this episode
    
    Constants derived from scene:
      - Cube height = 0.046m → stacked height diff ≈ 0.0468 (center-to-center)
      - Tolerance ±0.02m accounts for settling/physics jitter
      - Gripper open check: |finger_pos - 0.04| < 0.01 (relaxed from 1mm to 1cm
        for discoverability while still requiring intentional release)
    
    Validation:
      t=0: cubes side by side, z_diff ≈ 0 → |0 - 0.0468| = 0.0468 > 0.02 → fail ✓
      t=0: even if xy aligned, height_diff wrong → fail ✓
      gaming: stack without lifting → was_lifted=False → fail ✓
      gaming: hold cube above → gripper closed → fail ✓
    """
    _update_state(env)
    s = _get_state(env)

    robot: Articulation = env.scene[robot_cfg.name]
    cube1: RigidObject = env.scene[cube1_cfg.name]
    cube2: RigidObject = env.scene[cube2_cfg.name]

    pos_diff = cube1.data.root_pos_w - cube2.data.root_pos_w

    # 1. XY alignment
    xy_dist = torch.norm(pos_diff[:, :2], dim=1)
    xy_ok = xy_dist < xy_thresh

    # 2. Height difference matches one cube height
    z_diff = pos_diff[:, 2]
    height_ok = torch.abs(z_diff - height_diff) < height_tol

    # 3. Cube1 above cube2
    above_ok = z_diff > 0.0

    # 4. Gripper open (relaxed tolerance for discoverability)
    gripper_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    open_val = torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32, device=env.device)
    f0_open = torch.abs(robot.data.joint_pos[:, gripper_ids[0]] - open_val) < 0.01
    f1_open = torch.abs(robot.data.joint_pos[:, gripper_ids[1]] - open_val) < 0.01

    # 5. Was manipulated this episode
    success = xy_ok & height_ok & above_ok & f0_open & f1_open & s["was_lifted"]
    return success.float()


# ══════════════════════════════════════════════════════════════════════════════
# Reward 7: STABILITY BONUS — Reward sustained stacking
# ══════════════════════════════════════════════════════════════════════════════

def llm_stability(env) -> torch.Tensor:
    """Increasing reward for consecutive steps of successful stacking.
    
    Design rationale:
      - A momentary pass-through of the "stacked" state shouldn't give full reward
      - Counts consecutive steps where llm_stack_success=True
      - Reward scales linearly from 0→1 over 60 steps (~1 second at 60Hz)
      - Resets to 0 if stacking fails at any step
      - This solves the "transient pass-through" gaming problem
    
    Validation:
      t=0: not stacked → counter=0 → reward=0 ✓
      gaming: momentary alignment → counter reaches 1-2 → reward ≈ 0.02 (negligible) ✓
      success: stable stack for 1s → counter=60 → reward=1.0 ✓
    """
    s = _get_state(env)
    
    success = llm_stack_success(env)
    valid = success.bool() & s["was_lifted"]
    
    s["stable_steps"] = torch.where(valid, s["stable_steps"] + 1, torch.zeros_like(s["stable_steps"]))
    
    return (s["stable_steps"].float() / 60.0).clamp(0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# Reward 8: ACTION PENALTY — Smooth actions
# ══════════════════════════════════════════════════════════════════════════════

def llm_action_penalty(env) -> torch.Tensor:
    """L2 penalty on action rate (difference between consecutive actions).
    
    Design rationale:
      - Jerky motions waste energy and destabilize grasps
      - Small negative weight encourages smooth trajectories
      - Standard regularization, no gaming risk
    """
    return torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action),
        dim=1
    )


# ══════════════════════════════════════════════════════════════════════════════
# Curriculum — per-env difficulty progression
# ══════════════════════════════════════════════════════════════════════════════

def llm_curriculum(env, env_ids):
    """Per-env difficulty curriculum that advances on GOAL STATE achievement.
    
    d=0: cube2 near cube1 (easy transport)
    d=1: cube2 at default position (full task)
    
    Promotion-only until d=0.3 (prevent difficulty collapse in early training).
    """
    if not hasattr(llm_curriculum, "_n"):
        llm_curriculum._n = 0
        llm_curriculum._difficulty = torch.zeros(env.num_envs, device=env.device)
    llm_curriculum._n += 1

    cfg = _load_cfg()
    d = llm_curriculum._difficulty

    if isinstance(env_ids, torch.Tensor):
        ids = env_ids
    else:
        ids = (torch.tensor(env_ids, device=env.device, dtype=torch.long)
               if not isinstance(env_ids, slice)
               else torch.arange(env.num_envs, device=env.device))

    # Advance on GOAL STATE (full stack success + was_lifted)
    if ids.numel() > 0 and llm_curriculum._n > 2:
        with torch.no_grad():
            _update_state(env)
            s = _get_state(env)
            success = llm_stack_success(env).bool() & s["was_lifted"]

        step_up = cfg.get("step_up", 0.02)
        step_down = cfg.get("step_down", 0.003)
        promo_until = cfg.get("promotion_only_until", 0.3)
        promotion_only = d[ids] < promo_until
        demote = torch.where(promotion_only, torch.zeros_like(d[ids]),
                             torch.full_like(d[ids], step_down))
        delta = torch.where(success[ids], step_up, -demote)
        d[ids] = (d[ids] + delta).clamp(0.0, 1.0)

    # Hot-patch reward weights from config (every 50 curriculum calls)
    if llm_curriculum._n % 50 == 0:
        weights = cfg.get("reward_weights", {})
        for name, w in weights.items():
            try:
                idx = env.reward_manager._term_names.index(name)
                env.reward_manager._term_cfgs[idx].weight = w
            except (ValueError, IndexError):
                pass

    # Compute metrics for logging
    mean_d = d.mean().item()
    with torch.no_grad():
        reach_v = llm_reach(env).mean().item()
        grasp_v = llm_grasp(env).mean().item()
        lift_v = llm_lift(env).mean().item()
        transport_v = llm_transport(env).mean().item()
        stack_v = llm_stack_success(env).mean().item()

    return {
        "reaching": reach_v,
        "grasping": grasp_v,
        "lifting": lift_v,
        "above_cube2": transport_v,
        "stacking_success": stack_v,
        "mean_difficulty": mean_d,
        "d10": torch.quantile(d, 0.1).item(),
        "d90": torch.quantile(d, 0.9).item(),
        "config_version": cfg.get("version", 0),
    }


def llm_reset(env, env_ids, spawn_range: float = 0.03):
    """Curriculum-aware reset: randomize cube1, place cube2 by difficulty."""
    reset_scene_to_default(env, env_ids)
    _reset_state(env_ids)

    cfg = _load_cfg()
    sr = cfg.get("spawn_range", spawn_range)

    cube1: RigidObject = env.scene["cube_1"]
    cube2: RigidObject = env.scene["cube_2"]
    n = len(env_ids)

    # Cube1: small randomization around default
    c1 = cube1.data.default_root_state[env_ids].clone()
    for i in range(n):
        c1[i, 0] += torch.empty(1).uniform_(-sr, sr).item()
        c1[i, 1] += torch.empty(1).uniform_(-sr, sr).item()
    c1[:, 0:3] += env.scene.env_origins[env_ids]
    c1[:, 7:13] = 0
    cube1.write_root_pose_to_sim(c1[:, :7], env_ids=env_ids)
    cube1.write_root_velocity_to_sim(c1[:, 7:13], env_ids=env_ids)

    # Cube2: distance from cube1 controlled by per-env difficulty
    # d=0 → cube2 close to cube1 (min 5cm offset)
    # d=1 → cube2 at default distance
    c2 = cube2.data.default_root_state[env_ids].clone()
    tracker = getattr(llm_curriculum, "_difficulty", None)
    if tracker is not None:
        d = tracker[env_ids]
        c1_local = c1[:, :3] - env.scene.env_origins[env_ids]
        c2_default = c2[:, :3].clone()

        for i in range(n):
            di = d[i].item()
            dx = c2_default[i, 0] - c1_local[i, 0]
            dy = c2_default[i, 1] - c1_local[i, 1]
            full_dist = max((dx ** 2 + dy ** 2) ** 0.5, 1e-6)
            # Min 5cm offset to prevent physics overlap
            min_offset = 0.05
            effective_dist = min_offset + di * (full_dist - min_offset)
            c2[i, 0] = c1_local[i, 0] + effective_dist * dx / full_dist
            c2[i, 1] = c1_local[i, 1] + effective_dist * dy / full_dist
            # Small noise proportional to difficulty
            noise = 0.01 * max(di, 0.1)
            c2[i, 0] += torch.empty(1).uniform_(-noise, noise).item()
            c2[i, 1] += torch.empty(1).uniform_(-noise, noise).item()

    c2[:, 0:3] += env.scene.env_origins[env_ids]
    c2[:, 7:13] = 0
    cube2.write_root_pose_to_sim(c2[:, :7], env_ids=env_ids)
    cube2.write_root_velocity_to_sim(c2[:, 7:13], env_ids=env_ids)


# ══════════════════════════════════════════════════════════════════════════════
# Config classes
# ══════════════════════════════════════════════════════════════════════════════

@configclass
class LLMDesignedRewardsCfg:
    """Reward chain designed from first principles by LLM.
    
    Chain: reach → grasp → lift → transport → release → stack → stability
    All rewards written from scratch — no imports from mdp/rewards.py.
    """
    reaching = RewTerm(func=llm_reach, params={"std": 0.1}, weight=1.0)
    grasping = RewTerm(func=llm_grasp, params={"proximity_threshold": 0.06}, weight=5.0)
    lifting = RewTerm(func=llm_lift, params={"min_height": 0.06}, weight=3.0)
    above_cube2 = RewTerm(func=llm_transport, params={"std": 0.08, "min_height": 0.06}, weight=5.0)
    release = RewTerm(func=llm_release, params={"xy_thresh": 0.06, "min_height": 0.06}, weight=8.0)
    stacking = RewTerm(func=llm_stack_success, weight=25.0)
    stability = RewTerm(func=llm_stability, weight=15.0)
    action_rate = RewTerm(func=llm_action_penalty, weight=-0.1)


@configclass
class LLMDesignedCurriculumCfg:
    grasp = CurrTerm(func=llm_curriculum)


@configclass
class LLMDesignedEventCfg:
    reset = EventTerm(func=llm_reset, mode="reset", params={"spawn_range": 0.02})
