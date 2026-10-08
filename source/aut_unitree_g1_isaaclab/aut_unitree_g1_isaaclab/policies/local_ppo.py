"""Runs a locally trained rsl-rl >= 5 checkpoint as an inference policy."""

import torch


class LocalPPOPolicy:
    """Wraps an OnPolicyRunner-loaded actor for use by the dataset exporter."""

    def __init__(self, checkpoint_path: str, env, agent_cfg_dict: dict, device: str):
        from rsl_rl.runners import OnPolicyRunner

        self.env = env
        self.runner = OnPolicyRunner(
            env, agent_cfg_dict, log_dir=None, device=device
        )
        self.runner.load(str(checkpoint_path))
        self.policy = self.runner.get_inference_policy(device=device)

    @torch.no_grad()
    def __call__(self, obs):
        return self.policy(obs)
