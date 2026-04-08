---
name: curriculum-designer
description: Design and generate progressive training curricula for IsaacLab RL environments. Use when asked to create a curriculum for a robotics training task, design training stages, analyze an environment for curriculum opportunities, or compare curriculum approaches. Analyzes env configs, rewards, and physics to decompose tasks into learnable sub-skills with progressive difficulty scheduling. Generates Python curriculum code, registers new environments, and optionally runs training. Not for modifying existing ADR parameters, generic hyperparameter tuning, or non-IsaacLab environments.
---

# Curriculum Designer

Generate task-specific progressive training curricula for IsaacLab environments.

## Workflow

### 1. Analyze the target environment

Read these files for the target task (replace `<task_path>` with the actual path):
- `<task_path>/*_env_cfg.py` — env config (scene, rewards, events, commands, terminations)
- `<task_path>/mdp/rewards.py` — reward function implementations
- `<task_path>/mdp/terminations.py` — termination conditions
- `<task_path>/agents/rsl_rl_ppo_cfg.py` — training hyperparameters
- `<task_path>/config/<robot>/*_env_cfg.py` — robot-specific config

Read `references/methodology.md` for the full design methodology.

### 2. Design the curriculum stages

Based on task analysis, design 4-8 stages following the methodology:
- Identify sub-skills (reach → grasp → hold → lift → transport → place)
- Map difficulty axes (gravity, spawn range, goal range, noise)
- Set promotion conditions per stage

### 3. Generate the code

Create these files:
1. `<task_path>/<task>_llm_curriculum.py` — curriculum function + config classes
2. `<task_path>/config/<robot>/<robot>_llm_env_cfg.py` — env config override
3. Register the environment in `<task_path>/config/<robot>/__init__.py`

### 4. Run and validate

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task <registered_env_name> \
    --num_envs 4096 --max_iterations 500 --seed 42 --headless
```

Check: stages should consume ≥20% of total iterations in easy stages.

## Key lessons from experiments

- *Franka Lift*: 5-stage curriculum achieved 21x reward improvement over baseline
- *Dexsuite Kuka-Allegro*: Discrete stages promoted too fast (8 iters). Per-env continuous adaptation (ADR-style) is better for complex tasks
- *LLM v2 (per-env + smooth curves)*: 8.5x better success reward than ADR at 500 iters by using LLM-shaped parameter schedules (cubic gravity, sub-linear spawn widening)
- *Adaptive failure sampling*: Over-sample from spatial bins where the policy fails most. Combine with per-env difficulty for best results
- *Rule of thumb*: For simple tasks, global stages work great. For 23+ DoF dexterous tasks, combine LLM-designed curves + per-env pacing + failure-biased sampling
