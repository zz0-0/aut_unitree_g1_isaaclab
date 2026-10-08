"""Inference wrapper for classic rsl-rl checkpoints (pre-5.x schema)."""

import torch
from torch import nn


class ActorMLPPolicy:
    """Loads the actor MLP and observation normalizer from a classic rsl-rl checkpoint."""

    def __init__(self, model_state_dict: dict, device: str):
        weight_keys = sorted(
            (
                key
                for key in model_state_dict
                if key.startswith("actor.") and key.endswith(".weight")
            ),
            key=lambda key: int(key.split(".")[1]),
        )
        dims = [int(model_state_dict[key].shape[1]) for key in weight_keys]
        dims.append(int(model_state_dict[weight_keys[-1]].shape[0]))

        layers: list[nn.Module] = []
        for index in range(len(dims) - 1):
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            if index < len(dims) - 2:
                layers.append(nn.ELU())
        self.actor = nn.Sequential(*layers)
        actor_state = {
            key[len("actor.") :]: value
            for key, value in model_state_dict.items()
            if key.startswith("actor.") and "normalizer" not in key
        }
        self.actor.load_state_dict(actor_state)
        self.actor.to(device).eval()

        mean = model_state_dict.get("actor_obs_normalizer._mean")
        std = model_state_dict.get("actor_obs_normalizer._std")
        self.mean = None if mean is None else mean.to(device)
        self.std = None if std is None else std.to(device)
        self.eps = 1e-2

    @torch.no_grad()
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        features = obs
        if self.mean is not None and self.std is not None:
            features = (features - self.mean) / (self.std + self.eps)
        return self.actor(features)
