# Copyright (c) 2024, Curriculum Experiment
# Standalone cartpole training script for baseline vs curriculum comparison.

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train cartpole with RSL-RL (baseline vs curriculum).")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--task", type=str, default="Isaac-Cartpole-v0")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_iterations", type=int, default=150)
parser.add_argument("--experiment_name", type=str, default="cartpole_experiment")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import logging
import os
import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlVecEnvWrapper,
)

# Register our custom curriculum environment
import cartpole_curriculum_env_cfg  # noqa: F401

# Register standard cartpole
import isaaclab_tasks  # noqa: F401

logger = logging.getLogger(__name__)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    """Train cartpole and log results."""

    agent_cfg = RslRlOnPolicyRunnerCfg(
        num_steps_per_env=16,
        max_iterations=args_cli.max_iterations,
        save_interval=50,
        experiment_name=args_cli.experiment_name,
        seed=args_cli.seed,
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            actor_hidden_dims=[32, 32],
            critic_hidden_dims=[32, 32],
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
    )

    # Logging directory
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root_path, log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Resolve the env config from the gym registry
    spec = gym.spec(args_cli.task)
    env_cfg_entry = spec.kwargs.get("env_cfg_entry_point")
    if isinstance(env_cfg_entry, str):
        # Import the config class from the entry point string
        mod_name, cls_name = env_cfg_entry.rsplit(":", 1)
        import importlib
        mod = importlib.import_module(mod_name)
        env_cfg = getattr(mod, cls_name)()
    else:
        env_cfg = env_cfg_entry()

    # Override num_envs and seed
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    # Create environment with proper config
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    start_time = time.time()

    # Wrap for RSL-RL
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Create runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    # Dump configs
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # Run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    elapsed = time.time() - start_time

    # Save summary
    summary = {
        "experiment": args_cli.experiment_name,
        "task": args_cli.task,
        "training_time_s": round(elapsed, 2),
        "max_iterations": args_cli.max_iterations,
        "num_envs": args_cli.num_envs,
        "seed": args_cli.seed,
        "log_dir": log_dir,
    }

    summary_path = os.path.join(log_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {args_cli.experiment_name}")
    print(f"TASK: {args_cli.task}")
    print(f"Training time: {elapsed:.2f} seconds")
    print(f"Max iterations: {args_cli.max_iterations}")
    print(f"Log directory: {log_dir}")
    print(f"Summary saved: {summary_path}")
    print(f"{'='*60}\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
