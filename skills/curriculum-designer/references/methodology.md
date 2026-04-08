# Curriculum Design Methodology for Robot Learning
# A systematic approach to LLM-crafted progressive curricula

## Overview

This document defines a methodology for designing training curricula for
reinforcement learning tasks in robot simulation (e.g., IsaacLab). An LLM
analyzes the task structure and generates a progressive curriculum that
decomposes complex behaviors into learnable sub-skills.

## Step 1: Task Analysis

Read the following source files for the target environment:
- **Environment config** — scene, rewards, terminations, events, commands
- **Reward functions** — what behaviors are incentivized and their weights
- **Observation space** — what the policy can see
- **Action space** — what the policy controls
- **Existing curriculum** — any built-in difficulty scheduling

Identify:
1. **Sub-skills required** — decompose the task into ordered capabilities
2. **Difficulty axes** — which parameters make the task harder (gravity, spawn range, goal range, noise, object variety)
3. **Sparse reward bottlenecks** — where the agent gets stuck due to rare reward signals
4. **Physics coupling** — which parameters interact (e.g., gravity + grasping)

## Step 2: Stage Design

Design 4-8 explicit stages, each teaching ONE sub-skill:

### Rules for good stages:
- **One new challenge per stage** — don't increase gravity AND widen spawn range simultaneously
- **Start trivially easy** — the first stage should be almost impossible to fail
- **Each stage builds on the last** — sub-skills should compose
- **Final stage = baseline** — the last stage must exactly match the original task difficulty

### Parameters to schedule (by task type):

**Manipulation tasks:**
- Gravity: 0 → full (most impactful for grasping/lifting)
- Object spawn range: tight → full
- Goal position range: close → full
- Object variety: single shape → all shapes
- Joint noise: zero → full
- Observation noise: zero → full

**Locomotion tasks:**
- Terrain difficulty: flat → rough
- Command velocity range: slow → full
- Push force: zero → full
- Sensor noise: zero → full

## Step 3: Promotion Conditions

Each stage needs a task-specific metric for promotion:

### Guidelines:
- **Use the most relevant reward signal** for that stage's sub-skill
- **Set thresholds conservatively** — better to stay in a stage too long than promote too early
- **Minimum hold iterations** — enforce minimum time per stage (scale with task complexity)
- **Per-env vs global** — for >2048 envs, consider per-env difficulty (ADR-style)

### Recommended thresholds:
```
Simple task (Franka Lift):    min_hold = 50,  promote at 30% success
Medium task (Stack):          min_hold = 100, promote at 25% success  
Complex task (Dexterous):     min_hold = 200, promote at 20% success
```

### Critical lesson — pacing matters more than stage design:
With 4096 envs, even a small fraction succeeding is enough for promotion.
For complex tasks, prefer CONTINUOUS per-env scheduling over DISCRETE global stages.
The ADR pattern (per-env difficulty ±1 per episode) naturally provides a good curriculum
without requiring explicit stage thresholds.

## Step 4: Implementation Pattern

```python
# Standard curriculum function signature for IsaacLab:
def my_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
) -> dict[str, float]:
    # 1. Initialize state (first call)
    # 2. Record outcomes from ending episodes
    # 3. Check promotion condition
    # 4. Apply current stage settings (modify event_manager, command_manager)
    # 5. Return logged metrics
    ...
```

### What to modify per stage:
- `env.event_manager.cfg.<event>.params` — spawn ranges, gravity, etc.
- `env.command_manager.cfg.<command>.ranges` — goal ranges
- Reward weights via `mdp.modify_reward_weight`

## Step 5: Validation

After generating the curriculum:
1. Run 50-100 iterations and check stage progression timing
2. Verify the agent spends ≥20% of training in easy stages
3. Check that the final stage metrics match baseline difficulty
4. If stages promote too fast → increase min_hold or tighten thresholds
5. If stages never promote → relax thresholds or make easier stages

## Step 6: Adaptive Failure-Biased Sampling

In addition to progressive difficulty, bias the reset distribution toward
states where the policy fails most. This is "hard example mining" for RL.

### How it works:
1. **Bin the state space** — discretize initial conditions (object spawn x/y/z)
   into a spatial grid (e.g., 4×4×4 = 64 bins)
2. **Track outcomes per bin** — use an EMA of success rate per bin
3. **Sample proportionally to failure rate** — bins where the policy fails
   get more environments allocated to them at reset
4. **Exploration bonus** — bins with few samples get a boost to ensure coverage

### Implementation:
- At each reset, record which bin the env started in
- When the episode ends, update the bin's success EMA
- At the next reset, sample bin indices from `weights ∝ (1 - success_ema) + exploration_bonus`
- Sample positions uniformly within the chosen bin

### Key parameters:
```
ema_alpha = 0.05     # How fast to adapt (higher = more reactive)
num_bins = 4×4×4     # Spatial resolution (too many = sparse data)
exploration_bonus = 0.3 * (1 - confidence)  # Decays as bin gets more samples
min_weight = 0.05    # Floor to prevent any bin from being completely ignored
```

### When to use:
- Always beneficial for manipulation tasks (object spawn position matters)
- Most impactful when success varies significantly across the state space
- Combine with per-env difficulty for maximum effect

### When NOT to use:
- Locomotion tasks with uniform terrain (all states are equally hard)
- Very early in training (not enough data to estimate failure rates)
  → Use a warmup period of 2-3 reset cycles before activating

## Anti-patterns

- ❌ Promoting through all stages in <10% of training iterations
- ❌ Discrete gravity jumps (0 → 9.81) — use gradual ramps (0 → 3 → 6 → 9 → 9.81)
- ❌ Changing multiple difficulty axes simultaneously
- ❌ Global stage promotion with many envs (use per-env for >2048)
- ❌ No demotion — allow difficulty to decrease when performance drops
- ❌ Uniform sampling when failure rates vary across the state space
