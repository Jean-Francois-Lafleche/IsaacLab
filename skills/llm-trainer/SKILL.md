---
name: llm-trainer
description: LLM-in-the-loop RL training for IsaacLab. Decomposes tasks into skills, designs rewards and curricula, runs training in chunks, and adapts reward weights and curriculum parameters every N iterations based on training analysis. Use when training a complex manipulation or locomotion task where standard RL struggles. The LLM acts as an outer-loop optimizer that reads training metrics and adjusts the inner loop (rewards, curriculum, spawn ranges) to maximize learning efficiency.
---

# LLM Trainer — Adaptive Skill-Decomposed RL Training

## Overview

This skill implements an LLM-in-the-loop training system for IsaacLab RL tasks.
The LLM acts as an outer-loop optimizer that:
1. Decomposes the task into ordered skills
2. Designs reward functions and curriculum for each skill
3. Runs training in chunks (default: 100 iterations)
4. Analyzes training metrics after each chunk
5. Adjusts rewards, curriculum, and spawn ranges to maximize learning
6. Logs all decisions for transparency and reproducibility

## Workflow

### Step 1: Task Analysis
Read the target environment's config, rewards, scene, and observations.
Identify the skill decomposition (e.g., REACH → GRASP → LIFT → TRANSPORT → PLACE).

### Step 2: Initial Setup
Create:
- Reward functions for each skill (stored in a Python file)
- A curriculum config that reads parameters from `/tmp/llm_trainer_config.json`
- An env config that wires everything together
- Register the environment

### Step 3: Training Loop
For each chunk of N iterations:
1. Run training: `python train.py --task <env> --max_iterations <N>`
2. Read the training log and extract metrics
3. Analyze: which skill is the bottleneck? What's the success rate per skill?
4. Decide: adjust reward weights, curriculum parameters, spawn ranges
5. Write updated config to `/tmp/llm_trainer_config.json`
6. Log reasoning to `/tmp/llm_trainer_decisions.log`
7. Resume training from checkpoint

### Step 4: Analysis Template
After each chunk, answer these questions:
- Which skill phase is active? How long has it been active?
- What is the EMA reward for each skill?
- Is the current bottleneck skill improving, plateau'd, or declining?
- What would help: more reward signal? easier initial state? different weight balance?

### Decision Framework
```
IF skill_ema is INCREASING → keep current settings, don't interfere
IF skill_ema is PLATEAU'd for >50 iters → increase that skill's reward weight by 50%
IF skill_ema is DECLINING → check if spawn range is too hard, ease it
IF skill is MASTERED (ema > threshold) → activate next skill, reduce current weight by 30%
IF all skills active but final goal not achieved → boost final goal reward
```

## Config File Format
```json
{
    "version": 0,
    "active_skills": ["reach", "grasp"],
    "reward_weights": {"reach": 1.0, "grasp": 5.0, "lift": 0.0},
    "curriculum": {"spawn_range": 0.05, "step_up": 0.03},
    "llm_notes": "Iteration 100: grasp stuck at 0.01, boosting weight to 8.0"
}
```

## Key Lessons
- Binary rewards are too sparse for grasping — use distance-based shaping
- Reward weight changes can destabilize PPO if too sudden — change by max 50% per chunk
- The curriculum should match the active skill phase — no point widening spawn when still learning to grasp
- Log every decision so the human can understand and override
