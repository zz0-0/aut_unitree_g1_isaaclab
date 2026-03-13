# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import math
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ImuCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as base_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

##
# Pre-defined configs
##
from aut_unitree_g1_isaaclab.assets.robot.unitree import UNITREE_GO2_CFG


@configclass
class Go2Rewards(RewardsCfg):
    """Reward terms for GO2 quadruped locomotion.
    
    STABILITY IMPROVEMENTS:
    - Reduced termination penalty from -200.0 to -50.0 to prevent value explosion
    - Reduced flat_orientation penalty to prevent early training instability
    - Adjusted reward scales for better balance between positive and negative rewards
    """
    alive = RewTerm(func=base_mdp.is_alive, weight=1.0)
    termination_penalty = RewTerm(func=base_mdp.is_terminated, weight=-50.0)  # CHANGED: Reduced from -200.0

    track_lin_vel_xy_exp = RewTerm(
        func=base_mdp.track_lin_vel_xy_exp,
        weight=2.0,  # CHANGED: Increased from 1.5 to emphasize tracking
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=base_mdp.track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}  # CHANGED: Increased from 0.75
    )

    lin_vel_z_l2 = RewTerm(func=base_mdp.lin_vel_z_l2, weight=-1.0)  # CHANGED: Reduced from -2.0
    ang_vel_xy_l2 = RewTerm(func=base_mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=base_mdp.joint_torques_l2, weight=-0.0002)
    dof_acc_l2 = RewTerm(func=base_mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=base_mdp.action_rate_l2, weight=-0.01)
    dof_pos_limits = RewTerm(func=base_mdp.joint_pos_limits, weight=-1.0)

    # -- robot orientation
    flat_orientation_l2 = RewTerm(func=base_mdp.flat_orientation_l2, weight=-1.5)  # CHANGED: Reduced from -2.5

    # -- feet
    feet_air_time = RewTerm(
        func=base_mdp.feet_air_time,
        weight=0.05,  # CHANGED: Increased from 0.01 to encourage proper gait
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )


@configclass
class Go2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: Go2Rewards = Go2Rewards()
    
    def __post_init__(self):
        super().__post_init__()
        
        # Scene configuration
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        
        # Add IMU sensor on base
        self.scene.imu = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            update_period=0.0,  # Update at physics rate
            debug_vis=False,
        )
        
        # Scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # Reduce action scale for quadruped
        self.actions.joint_pos.scale = 0.25

        # Events - configure for quadruped with control group consistency
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        # Commands - disallow negative forward velocity (no backwards commands) for control consistency
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)

        # Terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"


@configclass
class Go2RoughEnvCfg_PLAY(Go2RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        
        # Spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        
        # Reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # Disable randomization for play
        self.observations.policy.enable_corruption = False
        
        # Remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        
        # Ensure the forward velocity range is non-negative for play mode as well
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
