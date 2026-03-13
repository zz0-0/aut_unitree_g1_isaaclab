# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class Go2RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for Unitree GO2 quadruped.
    
    This configuration is designed to match the G1 humanoid setup as closely as possible
    for control group comparison, while accounting for the quadruped's different morphology.
    
    STABILITY IMPROVEMENTS:
    - Enabled observation normalization to prevent value explosion
    - Reduced learning rate from 1e-3 to 5e-4 for more stable updates
    - Increased gradient clipping to prevent gradient explosion
    - Reduced entropy coefficient to prevent policy collapse
    - Using clipped value loss to bound critic updates
    """
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "go2_rough"
    
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,  # CHANGED: Enable to normalize inputs and prevent divergence
        critic_obs_normalization=True,  # CHANGED: Enable to stabilize value predictions
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,  # CHANGED: Reduced from 0.008 to prevent policy collapse
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,  # CHANGED: Reduced from 1e-3 to 5e-4 for stability
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=2.0,  # CHANGED: Increased from 1.0 to 2.0 for better gradient handling
    )


@configclass
class Go2FlatPPORunnerCfg(Go2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 1500
        self.experiment_name = "go2_flat"
        self.policy.actor_hidden_dims = [256, 128, 128]
        self.policy.critic_hidden_dims = [256, 128, 128]
