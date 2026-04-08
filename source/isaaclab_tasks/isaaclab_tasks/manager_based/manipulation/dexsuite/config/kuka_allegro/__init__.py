# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Dextra Kuka Allegro environments.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# State Observation
gym.register(
    id="Isaac-Dexsuite-Kuka-Allegro-Reorient-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_env_cfg:DexsuiteKukaAllegroReorientEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteKukaAllegroPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Dexsuite-Kuka-Allegro-Reorient-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_env_cfg:DexsuiteKukaAllegroReorientEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteKukaAllegroPPORunnerCfg",
    },
)

# Dexsuite Lift Environments
gym.register(
    id="Isaac-Dexsuite-Kuka-Allegro-Lift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_env_cfg:DexsuiteKukaAllegroLiftEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteKukaAllegroPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Dexsuite-Kuka-Allegro-Lift-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_env_cfg:DexsuiteKukaAllegroLiftEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteKukaAllegroPPORunnerCfg",
    },
)

# Baseline (no curriculum)
gym.register(
    id="Isaac-Dexsuite-Kuka-Allegro-Lift-Baseline-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_baseline_env_cfg:DexsuiteKukaAllegroLiftBaselineEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteKukaAllegroPPORunnerCfg",
    },
)

# Super Curriculum (ADR + adaptive failure sampling)
gym.register(
    id="Isaac-Dexsuite-Kuka-Allegro-Lift-SuperCurriculum-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_super_env_cfg:DexsuiteKukaAllegroLiftSuperCurriculumEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteKukaAllegroPPORunnerCfg",
    },
)

# LLM-Crafted Progressive Curriculum (no ADR)
gym.register(
    id="Isaac-Dexsuite-Kuka-Allegro-Lift-LLMCurriculum-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_llm_env_cfg:DexsuiteKukaAllegroLiftLLMCurriculumEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteKukaAllegroPPORunnerCfg",
    },
)

# LLM Curriculum v2 (per-env smooth interpolation)
gym.register(
    id="Isaac-Dexsuite-Kuka-Allegro-Lift-LLMv2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_llmv2_env_cfg:DexsuiteKukaAllegroLiftLLMv2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteKukaAllegroPPORunnerCfg",
    },
)

# LLM Curriculum v3 (per-env smooth + failure-biased sampling)
gym.register(
    id="Isaac-Dexsuite-Kuka-Allegro-Lift-LLMv3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_llmv3_env_cfg:DexsuiteKukaAllegroLiftLLMv3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteKukaAllegroPPORunnerCfg",
    },
)

# LLM Curriculum v3 optimized (8192 envs, larger network, faster ramp)
gym.register(
    id="Isaac-Dexsuite-Kuka-Allegro-Lift-LLMv3-Opt-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_llmv3_opt_env_cfg:DexsuiteKukaAllegroLiftLLMv3OptEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_optimized_ppo_cfg:DexsuiteOptimizedPPORunnerCfg",
    },
)

# LLM Curriculum v4 (sticky retry on failure)
gym.register(
    id="Isaac-Dexsuite-Kuka-Allegro-Lift-LLMv4-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_llmv4_env_cfg:DexsuiteKukaAllegroLiftLLMv4EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_optimized_ppo_cfg:DexsuiteOptimizedPPORunnerCfg",
    },
)
