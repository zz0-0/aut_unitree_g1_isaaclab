"""Runner for Unitree RL Lab exported ONNX velocity policies inside our IsaacLab env.

Replicates the observation group of ``unitree_rl_lab``'s
``Unitree-{G1-29dof,Go2}-Velocity`` tasks exactly:

    history_length = 5 of
    [base_ang_vel * 0.2, projected_gravity, velocity_commands,
     joint_pos_rel, joint_vel_rel * 0.05, last_action]

Each term's history (oldest first) is flattened and the terms are
concatenated in declaration order, matching IsaacLab's ObservationManager
with ``concatenate_terms=True``.
"""

from collections import deque
from pathlib import Path

import numpy as np
import torch

from isaaclab.utils.math import quat_apply_inverse

GRAVITY_VEC_W = (0.0, 0.0, -1.0)


class UnitreeRLLabOnnxPolicy:
    """Runs a unitree_rl_lab ONNX actor against a ManagerBasedRLEnv."""

    TERM_SPECS = (
        ("base_ang_vel", 0.2),
        ("projected_gravity", 1.0),
        ("velocity_commands", 1.0),
        ("joint_pos_rel", 1.0),
        ("joint_vel_rel", 0.05),
        ("last_action", 1.0),
    )

    def __init__(
        self,
        onnx_path: str | Path,
        env,
        action_scale: float = 0.25,
        history_length: int = 5,
        providers: list[str] | None = None,
    ):
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            str(onnx_path), providers=providers or ["CPUExecutionProvider"]
        )
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        batch_dim = input_meta.shape[0] if len(input_meta.shape) > 0 else None
        self.dynamic_batch = not isinstance(batch_dim, int)

        obs_dim = int(input_meta.shape[-1])
        per_step_dim = 9 + 3 * int(env.scene["robot"].num_joints)
        if obs_dim % per_step_dim == 0:
            history_length = max(1, obs_dim // per_step_dim)
        self.history_length = int(history_length)
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.action_scale = float(action_scale)
        self._histories = [deque(maxlen=self.history_length) for _ in self.TERM_SPECS]

        our_scale = self._resolve_our_action_scale(env)
        self._action_factor = (
            self.action_scale / our_scale
            if our_scale is not None and our_scale > 0
            else 1.0
        )

    @staticmethod
    def _resolve_our_action_scale(env) -> float | None:
        try:
            term = env.action_manager.get_term("joint_pos")
        except Exception:
            return None
        scale = getattr(term.cfg, "scale", None)
        if isinstance(scale, (int, float)):
            return float(scale)
        if torch.is_tensor(scale):
            return float(scale.mean().item())
        return None

    def _compute_terms(self) -> list[torch.Tensor]:
        robot = self.env.scene["robot"]
        quat = robot.data.root_quat_w
        gravity = torch.tensor(
            GRAVITY_VEC_W, device=quat.device, dtype=quat.dtype
        ).repeat(quat.shape[0], 1)
        terms = {
            "base_ang_vel": robot.data.root_ang_vel_b * 0.2,
            "projected_gravity": quat_apply_inverse(quat, gravity),
            "velocity_commands": self.env.command_manager.get_command("base_velocity"),
            "joint_pos_rel": robot.data.joint_pos - robot.data.default_joint_pos,
            "joint_vel_rel": robot.data.joint_vel * 0.05,
            "last_action": self.env.action_manager.action,
        }
        return [terms[name] for name, _ in self.TERM_SPECS]

    def _build_observation(self) -> np.ndarray:
        terms = self._compute_terms()
        if len(self._histories[0]) == 0:
            zeros = [torch.zeros_like(term) for term in terms]
            for _ in range(self.history_length - 1):
                for buffer, zero in zip(self._histories, zeros):
                    buffer.append(zero)
        for buffer, term in zip(self._histories, terms):
            buffer.append(term)
        observation = torch.cat(
            [torch.cat(list(buffer), dim=-1) for buffer in self._histories], dim=-1
        )
        return observation.detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def __call__(self, obs=None) -> torch.Tensor:
        observation = self._build_observation()
        if self.dynamic_batch:
            actions = self.session.run(None, {self.input_name: observation})[0]
        else:
            actions = np.concatenate(
                [
                    self.session.run(
                        None, {self.input_name: observation[index : index + 1]}
                    )[0]
                    for index in range(observation.shape[0])
                ],
                axis=0,
            )
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        return actions * self._action_factor
