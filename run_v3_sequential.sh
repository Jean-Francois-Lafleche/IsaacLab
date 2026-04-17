#!/bin/bash
set -e
cd /home/horde/IsaacLab
. /home/horde/isaaclab_env/bin/activate
export OMNI_KIT_ACCEPT_EULA=yes
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Stack-Cube-Franka-RL-GraspV2-v0 \
    --num_envs 4096 --max_iterations 5000 --seed 42 --headless
