#!/bin/bash
# Train all 3 stack curriculum variants sequentially: baseline, detailed, adaptive
# Each runs for 700 iterations with 4096 envs, seed 42, headless

set -e

export OMNI_KIT_ACCEPT_EULA=Y

PYTHON="/home/horde/isaaclab_env/bin/python"
ISAACLAB_DIR="/home/horde/IsaacLab"
TRAIN_SCRIPT="${ISAACLAB_DIR}/scripts/reinforcement_learning/rsl_rl/train.py"
COMMON_ARGS="--num_envs 4096 --seed 42 --headless --max_iterations 700"

echo "=============================================="
echo "STACK TRAINING: 3 CURRICULUM VARIANTS"
echo "=============================================="
echo ""

# 1. Baseline (no curriculum)
echo "[$(date '+%H:%M:%S')] Starting BASELINE training (Isaac-Stack-Cube-Franka-RL-v0)..."
cd "${ISAACLAB_DIR}/scripts/reinforcement_learning/rsl_rl"
${PYTHON} train.py --task Isaac-Stack-Cube-Franka-RL-v0 ${COMMON_ARGS} 2>&1 | tee /tmp/stack_baseline.log
echo "[$(date '+%H:%M:%S')] BASELINE training complete."
echo ""

# 2. Detailed static curriculum
echo "[$(date '+%H:%M:%S')] Starting DETAILED training (Isaac-Stack-Cube-Franka-RL-Detailed-v0)..."
${PYTHON} train.py --task Isaac-Stack-Cube-Franka-RL-Detailed-v0 ${COMMON_ARGS} 2>&1 | tee /tmp/stack_detailed.log
echo "[$(date '+%H:%M:%S')] DETAILED training complete."
echo ""

# 3. Adaptive LLM curriculum
echo "[$(date '+%H:%M:%S')] Starting ADAPTIVE training (Isaac-Stack-Cube-Franka-RL-Adaptive-v0)..."
${PYTHON} train.py --task Isaac-Stack-Cube-Franka-RL-Adaptive-v0 ${COMMON_ARGS} 2>&1 | tee /tmp/stack_adaptive.log
echo "[$(date '+%H:%M:%S')] ADAPTIVE training complete."
echo ""

echo "=============================================="
echo "ALL 3 TRAINING RUNS COMPLETE"
echo "Logs: /tmp/stack_{baseline,detailed,adaptive}.log"
echo "=============================================="
