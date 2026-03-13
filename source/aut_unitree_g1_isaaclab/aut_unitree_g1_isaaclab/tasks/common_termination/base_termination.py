import torch
from isaaclab.assets.articulation.articulation import Articulation
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv


def flag_reached_termination(
    env: ManagerBasedRLEnv, 
    flag_cfg: SceneEntityCfg = SceneEntityCfg("flag"), 
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_threshold: float = 0.08,  # 8cm - successful grasp distance
    hold_time_steps: int = 10,  # Must hold for 10 steps to confirm success
) -> torch.Tensor:
    """Terminate episode when hand reaches flag and holds position.
    
    This helps training by:
    1. Resetting faster when successful (more efficient data collection)
    2. Providing clear success signal
    3. Preventing wasted steps after reaching goal
    
    Args:
        distance_threshold: Distance in meters to consider "reached"
        hold_time_steps: Number of consecutive steps within threshold to confirm success
    """
    flag: RigidObject = env.scene[flag_cfg.name]
    flag_pos = flag.data.root_pos_w

    robot: Articulation = env.scene[robot_cfg.name]
    left_idx = robot.data.body_names.index("left_hand_base_link")
    right_idx = robot.data.body_names.index("right_hand_base_link")
    
    left_hand_pos = robot.data.body_pos_w[:, left_idx]
    right_hand_pos = robot.data.body_pos_w[:, right_idx]
    
    # Calculate distances
    left_dist = torch.norm(flag_pos - left_hand_pos, dim=1)
    right_dist = torch.norm(flag_pos - right_hand_pos, dim=1)
    min_dist = torch.min(left_dist, right_dist)
    
    # Check if within threshold
    within_threshold = min_dist < distance_threshold
    
    # Track consecutive steps within threshold
    if not hasattr(env, '_success_counter'):
        env._success_counter = torch.zeros(env.num_envs, dtype=torch.int32, device=min_dist.device)
    
    # Update counter
    env._success_counter[within_threshold] += 1
    env._success_counter[~within_threshold] = 0
    
    # Reset counter on episode reset
    if hasattr(env, 'episode_length_buf'):
        reset_mask = env.episode_length_buf == 0
        env._success_counter[reset_mask] = 0
    
    # Terminate if held for required time
    success = env._success_counter >= hold_time_steps
    
    # Debug print when success occurs
    if success.any():
        for idx in torch.where(success)[0]:
            print(f"[SUCCESS] Env {idx}: Flag reached! Distance={min_dist[idx]:.3f}m, Held for {env._success_counter[idx]} steps")
    
    return success


def reset_object_estimate(env: ManagerBasedRLEnv, flag_cfg: SceneEntityCfg = SceneEntityCfg("flag"), robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    flag: RigidObject = env.scene[flag_cfg.name]
    flag_pos = flag.data.root_pos_w

    robot : Articulation = env.scene[robot_cfg.name]
    left_idx  = robot.data.body_names.index("left_hand_base_link")
    right_idx = robot.data.body_names.index("right_hand_base_link")
    left_hand_pos = robot.data.body_pos_w[:, left_idx]
    right_hand_pos = robot.data.body_pos_w[:, right_idx]

    flag_x = flag_pos[:, 0]
    flag_y = flag_pos[:, 1]
    flag_z = flag_pos[:, 2]

    left_min_x = left_hand_pos[:, 0] - 0.20
    left_max_x = left_hand_pos[:, 0] + 0.20
    left_min_y = left_hand_pos[:, 1] - 0.20
    left_max_y = left_hand_pos[:, 1] + 0.20
    left_min_z = left_hand_pos[:, 2] - 0.20
    left_max_z = left_hand_pos[:, 2] + 0.20
    right_min_x = right_hand_pos[:, 0] - 0.20
    right_max_x = right_hand_pos[:, 0] + 0.20
    right_min_y = right_hand_pos[:, 1] - 0.20
    right_max_y = right_hand_pos[:, 1] + 0.20
    right_min_z = right_hand_pos[:, 2] - 0.20
    right_max_z = right_hand_pos[:, 2] + 0.20

    done_x = ((flag_x >= left_min_x) and (flag_x <= left_max_x)) or ((flag_x >= right_min_x) and (flag_x <= right_max_x))
    done_y = ((flag_y >= left_min_y) and (flag_y <= left_max_y)) or ((flag_y >= right_min_y) and (flag_y <= right_max_y))
    done_z = ((flag_z >= left_min_z) and (flag_z <= left_max_z)) or ((flag_z >= right_min_z) and (flag_z <= right_max_z))
    done = done_x and done_y and done_z
    return not done