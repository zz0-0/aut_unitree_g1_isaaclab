from isaaclab.assets.articulation.articulation import Articulation
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply
import torch
import sys
import os

_rewards_dds = None
_dds_initialized = False

def _get_rewards_dds_instance():
    """get the DDS instance, delay initialization"""
    global _rewards_dds, _dds_initialized
    
    if not _dds_initialized or _rewards_dds is None:
        try:
            # dynamically import the DDS module
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dds'))
            from dds.dds_master import dds_manager
            
            _rewards_dds = dds_manager.get_object("rewards")
            print("[Observations Rewards] DDS communication instance obtained")
            
            # register the cleanup function
            import atexit
            def cleanup_dds():
                try:
                    if _rewards_dds:
                        dds_manager.unregister_object("rewards")
                        print("[rewards_dds] DDS communication closed correctly")
                except Exception as e:
                    print(f"[rewards_dds] Error closing DDS: {e}")
            atexit.register(cleanup_dds)
            
        except Exception as e:
            print(f"[Observations Rewards] Failed to get DDS instances: {e}")
            _rewards_dds = None
        
        _dds_initialized = True
    
    return _rewards_dds

def flag_distance_to_hands(
        env: ManagerBasedRLEnv,
        flag_cfg: SceneEntityCfg = SceneEntityCfg("flag"),
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    ✅ DISTANCE-BASED REWARD - Provides smooth gradient for learning
    
    Key improvements:
    1. Smooth reward signal at ALL distances (not just binary)
    2. Hand orientation offset for accurate palm position
    3. Exponential shaping for stronger signal when close
    """
    
    flag: RigidObject = env.scene[flag_cfg.name]
    flag_pos = flag.data.root_pos_w

    robot: Articulation = env.scene[robot_cfg.name]

    # Get hand BASE positions (wrist)
    left_idx  = robot.data.body_names.index("left_hand_base_link")
    right_idx = robot.data.body_names.index("right_hand_base_link")

    left_hand_pos = robot.data.body_pos_w[:, left_idx]
    right_hand_pos = robot.data.body_pos_w[:, right_idx]
    
    # ✅ CRITICAL FIX: Add hand offset to get palm/fingertip position
    # The hand extends ~0.20m forward from wrist in the hand's forward direction
    hand_forward_offset = 0.20  # 20cm from wrist to effective grasp point
    
    # Get hand orientations (quaternions)
    left_hand_quat = robot.data.body_quat_w[:, left_idx]   # (N, 4) - quaternion [w, x, y, z]
    right_hand_quat = robot.data.body_quat_w[:, right_idx]
    
    # Define forward direction in hand's local frame (assume x-axis is forward)
    local_forward = torch.tensor([hand_forward_offset, 0.0, 0.0], device=left_hand_pos.device)
    local_forward = local_forward.unsqueeze(0).expand(left_hand_pos.shape[0], -1)
    
    # Rotate the local forward vector by the hand's orientation to get world-frame offset
    left_palm_offset = quat_apply(left_hand_quat, local_forward)
    right_palm_offset = quat_apply(right_hand_quat, local_forward)
    
    # Add offset to get palm positions
    left_palm_pos = left_hand_pos + left_palm_offset
    right_palm_pos = right_hand_pos + right_palm_offset

    left_dist = torch.norm(flag_pos - left_palm_pos, dim=1)
    right_dist = torch.norm(flag_pos - right_palm_pos, dim=1)

    # ✅ BOTH HANDS REWARD - Calculate reward for EACH hand independently, then combine
    # This encourages BOTH hands to reach toward the flag, not just pick the closer one
    
    # Max distance we care about: 2.0m
    max_distance = 2.0
    linear_scale = 15.0  # Higher weight for linear component
    exp_scale = 2.5  # Tuned for 0.5m range focus
    
    # === LEFT HAND REWARD ===
    left_linear = torch.clamp(linear_scale * (1.0 - left_dist / max_distance), min=0.0)
    left_exp = 25.0 * torch.exp(-exp_scale * left_dist)
    
    # Left hand bonuses
    left_bonus = torch.zeros_like(left_dist)
    left_bonus[left_dist < 0.50] += 5.0
    left_bonus[left_dist < 0.30] += 8.0
    left_bonus[left_dist < 0.20] += 12.0
    left_bonus[left_dist < 0.10] += 20.0
    left_bonus[left_dist < 0.05] += 40.0
    
    left_reward = left_linear + left_exp + left_bonus
    
    # === RIGHT HAND REWARD ===
    right_linear = torch.clamp(linear_scale * (1.0 - right_dist / max_distance), min=0.0)
    right_exp = 25.0 * torch.exp(-exp_scale * right_dist)
    
    # Right hand bonuses
    right_bonus = torch.zeros_like(right_dist)
    right_bonus[right_dist < 0.50] += 5.0
    right_bonus[right_dist < 0.30] += 8.0
    right_bonus[right_dist < 0.20] += 12.0
    right_bonus[right_dist < 0.10] += 20.0
    right_bonus[right_dist < 0.05] += 40.0
    
    right_reward = right_linear + right_exp + right_bonus
    
    # ✅ COMBINE BOTH HANDS - Sum rewards (not min/max!)
    # This way, BOTH hands contribute to the total reward
    reward = left_reward + right_reward
    
    # For progress tracking, use the minimum distance (closest hand)
    min_dist = torch.min(left_dist, right_dist)
    
    # Component 4: Progress reward (encourage continuous improvement)
    # Track best distance achieved this episode (using closest hand)
    if not hasattr(env, '_best_distance'):
        env._best_distance = torch.full((env.num_envs,), float('inf'), device=min_dist.device)
    
    # Reset best distance on episode reset
    if hasattr(env, 'episode_length_buf'):
        reset_mask = env.episode_length_buf == 0
        env._best_distance[reset_mask] = float('inf')
    
    # Reward for achieving new best distance
    improvement = env._best_distance - min_dist
    progress_reward = torch.clamp(improvement * 20.0, min=0.0, max=10.0)  # Up to 10 bonus for improvement
    env._best_distance = torch.min(env._best_distance, min_dist)
    
    reward += progress_reward

    # Debug: Print comprehensive information every 50 steps
    if hasattr(env, 'episode_length_buf') and env.episode_length_buf[0] % 50 == 0:
        env_idx = 0
        print(f"[REWARD] Step {env.episode_length_buf[env_idx].item():3d}: "
              f"Dist L:{left_dist[env_idx]:.3f}m R:{right_dist[env_idx]:.3f}m (min:{min_dist[env_idx]:.3f}), "
              f"Reward={reward[env_idx]:.2f} (L:{left_reward[env_idx]:.1f} R:{right_reward[env_idx]:.1f} "
              f"prog:{progress_reward[env_idx]:.1f}), "
              f"Best={env._best_distance[env_idx]:.3f}m")

    return reward