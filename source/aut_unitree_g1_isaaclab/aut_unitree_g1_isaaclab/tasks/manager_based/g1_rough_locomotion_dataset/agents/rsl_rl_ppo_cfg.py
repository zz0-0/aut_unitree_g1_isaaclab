# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class G1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for Unitree G1 humanoid.
    
    This configuration matches the GO2 setup for control group comparison.
    
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
    experiment_name = "g1_rough"
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
        entropy_coef=0.005,  # CHANGED: Reduced to 0.005 to prevent policy collapse (matches GO2)
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,  # CHANGED: Reduced from 1e-3 to 5e-4 for stability (matches GO2)
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=2.0,  # CHANGED: Increased from 1.0 to 2.0 for better gradient handling (matches GO2)
    )


@configclass
class G1FlatPPORunnerCfg(G1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 1500
        self.experiment_name = "g1_flat"
        self.policy.actor_hidden_dims = [256, 128, 128]
        self.policy.critic_hidden_dims = [256, 128, 128]
