import torch

from isaaclab.envs import ManagerBasedRLEnv

def get_flag_position(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Get the position of the flag object in the environment.

    Args:
        env (ManagerBasedRLEnv): The environment instance.

    Returns:
        torch.Tensor: A tensor containing the position of the flag.
    """
    flag_entity = env.scene["flag"]
    if flag_entity is None:
        raise ValueError("Flag entity not found in the environment scene.")

    flag_position = flag_entity.data.root_pos_w
    return flag_position