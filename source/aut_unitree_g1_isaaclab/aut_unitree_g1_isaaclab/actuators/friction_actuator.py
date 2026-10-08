"""Actuator models with an explicit joint friction torque.

PhysX applies joint friction internally and does not expose the resulting
friction torque. To obtain an exactly known proprioceptive target for joint
friction estimation, these actuators inject an explicit Coulomb + viscous
friction torque as a feedforward actuation effort and expose the injected
torque through ``friction_torque`` for dataset logging.

The friction torque applied to joint ``j`` of environment ``i`` is::

    tau_friction[i, j] = -(mu[i, j] * sign(qd[i, j]) + c_v[i, j] * qd[i, j])

where ``mu`` (Coulomb) and ``c_v`` (viscous) are sampled once per environment
from the configured ranges, so a state estimator must infer them from
proprioception instead of reading them directly.
"""

from __future__ import annotations

import torch

from isaaclab.actuators import DCMotor, ImplicitActuator
from isaaclab.actuators.actuator_cfg import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class FrictionTorqueMixin:
    """Injects an explicit Coulomb + viscous friction torque into an actuator model."""

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._init_friction_model(cfg)

    def _init_friction_model(self, cfg) -> None:
        mu_low, mu_high = getattr(cfg, "friction_mu_range", (0.0, 0.0))
        viscous_low, viscous_high = getattr(cfg, "friction_viscous_range", (0.0, 0.0))
        seed = int(getattr(cfg, "friction_seed", 0))
        seed += sum(ord(char) for char in "".join(self._joint_names)) % 100_000

        generator = torch.Generator(device="cpu").manual_seed(seed)
        shape = (self._num_envs, self.num_joints)
        self.friction_mu = (
            torch.empty(shape)
            .uniform_(float(mu_low), float(mu_high), generator=generator)
            .to(self._device)
        )
        self.friction_viscous = (
            torch.empty(shape)
            .uniform_(float(viscous_low), float(viscous_high), generator=generator)
            .to(self._device)
        )
        self.friction_torque = torch.zeros(shape, device=self._device)
        self.friction_enabled = any(
            value > 0.0 for value in (mu_low, mu_high, viscous_low, viscous_high)
        )

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        control_action = super().compute(control_action, joint_pos, joint_vel)

        friction = -(
            self.friction_mu * torch.sign(joint_vel)
            + self.friction_viscous * joint_vel
        )
        self.friction_torque[:] = friction
        if self.friction_enabled:
            if control_action.joint_efforts is None:
                control_action.joint_efforts = friction
            else:
                control_action.joint_efforts = control_action.joint_efforts + friction
        return control_action


class FrictionImplicitActuator(FrictionTorqueMixin, ImplicitActuator):
    """Implicit PD actuator with explicit friction torque injection."""


class FrictionDCMotor(FrictionTorqueMixin, DCMotor):
    """DC motor actuator with explicit friction torque injection."""


@configclass
class FrictionImplicitActuatorCfg(ImplicitActuatorCfg):
    """Configuration for :class:`FrictionImplicitActuator`."""

    class_type: type = FrictionImplicitActuator

    friction_mu_range: tuple[float, float] = (0.0, 0.0)
    """Per-environment Coulomb friction torque range in N*m."""

    friction_viscous_range: tuple[float, float] = (0.0, 0.0)
    """Per-environment viscous friction coefficient range in N*m*s/rad."""

    friction_seed: int = 0
    """Seed used to sample per-environment friction coefficients."""


@configclass
class FrictionDCMotorCfg(DCMotorCfg):
    """Configuration for :class:`FrictionDCMotor`."""

    class_type: type = FrictionDCMotor

    friction_mu_range: tuple[float, float] = (0.0, 0.0)
    """Per-environment Coulomb friction torque range in N*m."""

    friction_viscous_range: tuple[float, float] = (0.0, 0.0)
    """Per-environment viscous friction coefficient range in N*m*s/rad."""

    friction_seed: int = 0
    """Seed used to sample per-environment friction coefficients."""
