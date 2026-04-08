# Optimized PPO config for faster convergence on L40 GPU

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class DexsuiteOptimizedPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    obs_groups = {"policy": ["policy", "proprio", "perception"], "critic": ["policy", "proprio", "perception"]}
    max_iterations = 15000
    save_interval = 250
    experiment_name = "dexsuite_kuka_allegro"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 512, 256],    # Larger network
        critic_hidden_dims=[512, 512, 256],    # Larger network
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=8,    # 8 instead of 4 → better gradient estimates
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
