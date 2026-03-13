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
import aut_unitree_g1_isaaclab.tasks.manager_based.g1_rough_locomotion_dataset.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

##
# Pre-defined configs
##
# from isaaclab_assets import G1_MINIMAL_CFG  # isort: skip
from aut_unitree_g1_isaaclab.tasks.common_config import G1RobotPresets


@configclass
class G1Rewards(RewardsCfg):
    """Reward terms for G1 humanoid locomotion.
    
    STABILITY IMPROVEMENTS (matching GO2):
    - Reduced termination penalty from -200.0 to -50.0 to prevent value explosion
    - Increased positive tracking rewards to balance negatives
    - Reduced several penalty weights for more stable training
    - Better balance between positive and negative rewards
    """
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-50.0)  # CHANGED: Reduced from -200.0 (matches GO2)

    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,  # CHANGED: Increased from 1.0 to emphasize tracking (proportional to GO2)
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}  # CHANGED: Increased from 0.5 (proportional to GO2)
    )

    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)  # CHANGED: Reduced from -2.0 (matches GO2)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
            )
        },
    )
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist.*",
                ],
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    # -- robot
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-3.0)  # CHANGED: Reduced from -5.0 (proportional to GO2)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-6.0, params={"target_height": 0.78})  # CHANGED: Reduced from -10 for stability

    # -- feet
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.5,
        params={
            "period": 0.8,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=1.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )

    # -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )
    # Penalize actual backward motion (base frame negative X velocity)
    backward_motion_penalty = RewTerm(func=mdp.backward_motion_penalty, weight=-1.5)

@configclass
class G1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: G1Rewards = G1Rewards()
    
    def __post_init__(self):
        super().__post_init__()
        # NOTE: init_rot must be a valid quaternion (w, x, y, z) with non-zero norm.
        # Using an all-zero quaternion is invalid and can cause PhysX errors
        # because transforms and derived inertia tensors will be invalid.
        self.scene.robot = G1RobotPresets.g1_29dof_inspire_wholebody_cfg(
            init_pos=(-0.15, 0.0, 0.744),
            init_rot=(0.7071, 0, 0, 0.7071)  # identity quaternion (w, x, y, z)
        ).replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        
        # Add IMU sensor on torso_link
        self.scene.imu = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            update_period=0.0,  # Update at physics rate
            debug_vis=False,
        )
        
        self.events.add_base_mass = None
        # disallow negative forward velocity commands (no backwards commands)
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)

    # def __post_init__(self):
    #     # post init of parent
    #     super().__post_init__()
    #     # Scene
    #     self.scene.robot = G1RobotPresets.g1_29dof_inspire_wholebody_cfg(
    #         init_pos=(0, 0.0, 0.76),
    #         init_rot=(1, 0, 0, 0)
    #     ).replace(prim_path="{ENV_REGEX_NS}/Robot")
    #     self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

    #     # Randomization
    #     self.events.push_robot = None
    #     self.events.add_base_mass = None
    #     self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    #     self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
    #     self.events.reset_base.params = {
    #         "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
    #         "velocity_range": {
    #             "x": (0.0, 0.0),
    #             "y": (0.0, 0.0),
    #             "z": (0.0, 0.0),
    #             "roll": (0.0, 0.0),
    #             "pitch": (0.0, 0.0),
    #             "yaw": (0.0, 0.0),
    #         },
    #     }
    #     self.events.base_com = None

    #     # Rewards
    #     self.rewards.lin_vel_z_l2.weight = 0.0
    #     self.rewards.undesired_contacts = None
    #     self.rewards.flat_orientation_l2.weight = -1.0
    #     self.rewards.action_rate_l2.weight = -0.005
    #     self.rewards.dof_acc_l2.weight = -1.25e-7
    #     self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
    #         "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
    #     )
    #     self.rewards.dof_torques_l2.weight = -1.5e-7
    #     self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
    #         "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
    #     )

    #     # Commands
    #     self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
    #     self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
    #     self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

    #     # terminations
    #     self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"


@configclass
class G1RoughEnvCfg_PLAY(G1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        # Ensure the forward velocity range is non-negative for play mode as well
        # (limit_ranges may overwrite the dataset-specific range set above)
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)

