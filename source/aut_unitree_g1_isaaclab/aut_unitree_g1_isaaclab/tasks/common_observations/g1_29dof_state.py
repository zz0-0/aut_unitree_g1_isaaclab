# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""
g1_29dof state
"""     
from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import os

import sys
from multiprocessing import shared_memory

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


import torch

def get_robot_boy_joint_names() -> list[str]:
    return [
        # leg joints (12)
        # left leg (6)
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        # right leg (6)
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        # waist joints (3)
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",

        # arm joints (14)
        # left arm (7)
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        # right arm (7)
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]

def get_robot_arm_joint_names() -> list[str]:
    return [
        # arm joints (14)
        # left arm (7)
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        # right arm (7)
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]

# global variable to cache the DDS instance
_g1_robot_dds = None
_dds_initialized = False

# 观测缓存：索引张量与DDS限速（50FPS）+ 预分配缓冲
_obs_cache = {
    "device": None,
    "batch": None,
    "boy_idx_t": None,
    "boy_idx_batch": None,
    "pos_buf": None,
    "vel_buf": None,
    "torque_buf": None,
    "combined_buf": None,
    "dds_last_ms": 0,
    "dds_min_interval_ms": 20,
}

def _get_g1_robot_dds_instance():
    """get the DDS instance, delay initialization"""
    global _g1_robot_dds, _dds_initialized
    
    if not _dds_initialized or _g1_robot_dds is None:
        try:
            # dynamically import the DDS module
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dds'))
            from dds.dds_master import dds_manager
            print(f"dds_manager: {dds_manager}")
            _g1_robot_dds = dds_manager.get_object("g129")
            print("[g1_state] G1 robot DDS communication instance obtained")
            
            # register the cleanup function
            import atexit
            def cleanup_dds():
                try:
                    if _g1_robot_dds:
                        dds_manager.unregister_object("g129")
                        print("[g1_state] DDS communication closed correctly")
                except Exception as e:
                    print(f"[g1_state] Error closing DDS: {e}")
            atexit.register(cleanup_dds)
            
        except Exception as e:
            print(f"[g1_state] Failed to get G1 robot DDS instance: {e}")
            _g1_robot_dds = None
        
        _dds_initialized = True
    
    return _g1_robot_dds

def get_robot_body_joint_states(
    env: ManagerBasedRLEnv,
    enable_dds: bool = False,
) -> torch.Tensor:
    """get the robot body joint states, positions and velocities
    
    Args:
        env: ManagerBasedRLEnv - reinforcement learning environment instance
        enable_dds: bool - whether to enable the DDS publish function
    
    Returns:
        torch.Tensor
        - the first 29 elements are joint positions
        - the middle 29 elements are joint velocities
        - the last 29 elements are joint torques
    """
    # get all joint states
    joint_pos = env.scene["robot"].data.joint_pos
    joint_vel = env.scene["robot"].data.joint_vel
    joint_torque = env.scene["robot"].data.applied_torque  # use applied_torque to get joint torques
    device = joint_pos.device
    batch = joint_pos.shape[0]

    # 预计算并缓存索引张量（列索引）
    global _obs_cache
    if _obs_cache["device"] != device or _obs_cache["boy_idx_t"] is None:
        boy_joint_indices = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
        _obs_cache["boy_idx_t"] = torch.tensor(boy_joint_indices, dtype=torch.long, device=device)
        _obs_cache["device"] = device
        _obs_cache["batch"] = None  # force re-init batch-shaped buffers

    idx_t = _obs_cache["boy_idx_t"]
    n = idx_t.numel()

    # 预分配/复用 batch 形状索引与输出缓冲
    if _obs_cache["batch"] != batch or _obs_cache["boy_idx_batch"] is None:
        _obs_cache["boy_idx_batch"] = idx_t.unsqueeze(0).expand(batch, n)
        _obs_cache["pos_buf"] = torch.empty(batch, n, device=device, dtype=joint_pos.dtype)
        _obs_cache["vel_buf"] = torch.empty(batch, n, device=device, dtype=joint_pos.dtype)
        _obs_cache["torque_buf"] = torch.empty(batch, n, device=device, dtype=joint_pos.dtype)
        _obs_cache["combined_buf"] = torch.empty(batch, n * 3, device=device, dtype=joint_pos.dtype)
        _obs_cache["batch"] = batch

    idx_batch = _obs_cache["boy_idx_batch"]
    pos_buf = _obs_cache["pos_buf"]
    vel_buf = _obs_cache["vel_buf"]
    torque_buf = _obs_cache["torque_buf"]
    combined_buf = _obs_cache["combined_buf"]

    # 使用 gather(out=...) 填充，避免新张量分配
    try:
        torch.gather(joint_pos, 1, idx_batch, out=pos_buf)
        torch.gather(joint_vel, 1, idx_batch, out=vel_buf)
        torch.gather(joint_torque, 1, idx_batch, out=torque_buf)
    except TypeError:
        pos_buf.copy_(torch.gather(joint_pos, 1, idx_batch))
        vel_buf.copy_(torch.gather(joint_vel, 1, idx_batch))
        torque_buf.copy_(torch.gather(joint_torque, 1, idx_batch))

    # 组合为一个缓冲，避免 cat 分配
    combined_buf[:, 0:n].copy_(pos_buf)
    combined_buf[:, n:2*n].copy_(vel_buf)
    combined_buf[:, 2*n:3*n].copy_(torque_buf)

    # write to DDS（限速发布，避免高频CPU拷贝）
    if enable_dds and combined_buf.shape[0] > 0:
        try:
            import time
            now_ms = int(time.time() * 1000)
            if now_ms - _obs_cache["dds_last_ms"] >= _obs_cache["dds_min_interval_ms"]:
                g1_robot_dds = _get_g1_robot_dds_instance()
                if g1_robot_dds:
                    imu_data = get_robot_imu_data(env)
                    if imu_data.shape[0] > 0:
                        g1_robot_dds.write_robot_state(
                            pos_buf[0].contiguous().cpu().numpy(),
                            vel_buf[0].contiguous().cpu().numpy(),
                            torque_buf[0].contiguous().cpu().numpy(),
                            imu_data[0].contiguous().cpu().numpy(),
                        )
                        _obs_cache["dds_last_ms"] = now_ms
        except Exception as e:
            print(f"[g1_state] Error writing robot state to DDS: {e}")
    
    return combined_buf


def get_gravity_quaternion_from_root_state(env: ManagerBasedRLEnv):
    body_names = env.scene["robot"].data.body_names
    imu_pelvis_idx = body_names.index("imu_in_pelvis")
    imu_torso_idx = body_names.index("imu_in_torso")
    print(f"imu_pelvis_idx: {imu_pelvis_idx}")
    print(f"imu_torso_idx: {imu_torso_idx}")
    pose = env.scene["robot"].data.body_link_pose_w  # [num_links, 7] (pos + quat)
    vel = env.scene["robot"].data.body_link_vel_w    # [num_links, 6] (lin_vel + ang_vel)

    # 取出IMU位置+旋转
    pelvis_pose = pose[:, imu_pelvis_idx, :]  # [B, 7]
    torso_pose = pose[:, imu_torso_idx, :]

    # 取出IMU线速度+角速度
    pelvis_vel = vel[:, imu_pelvis_idx, :]    # [B, 6]
    torso_vel = vel[:, imu_torso_idx, :]
    return pelvis_pose, pelvis_vel, torso_pose, torso_vel

def get_robot_imu_data(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """get the robot IMU data
    
    Args:
        env: ManagerBasedRLEnv - reinforcement learning environment instance
    
    Returns:
        torch.Tensor
        - the first 3 elements are position
        - the next 4 elements are rotation quaternion
        - the next 3 elements are linear velocity
        - the last 3 elements are angular velocity
    """
    # get the robot root state
    root_state = env.scene["robot"].data.root_state_w
    # print(env.scene["robot"].data.__dict__.keys())
    # pelvis_pose, pelvis_vel, torso_pose, torso_vel = get_gravity_quaternion_from_root_state(env)
    # print(f"pelvis_pose: {pelvis_pose}")
    # print(f"pelvis_vel: {pelvis_vel}")
    # print(f"torso_pose: {torso_pose}")
    # print(f"torso_vel: {torso_vel}")
    # extract the position, rotation, velocity and angular velocity
    pos = root_state[:, :3]  # position
    quat = root_state[:, 3:7]  # rotation quaternion
    vel = root_state[:, 7:10]  # linear velocity
    ang_vel = root_state[:, 10:13]  # angular velocity
    # pos = torso_pose[:, :3]
    # quat = torso_pose[:, 3:7]
    # vel = torso_vel[:, :3]
    # ang_vel = torso_vel[:, 3:6]
    # qu = get_gravity_quaternion(env)
    # print(f"qu: {qu}")
    # print(f"quat: {quat}")
    # concatenate all data
    imu_data = torch.cat([pos, quat, vel, ang_vel], dim=1)
    
    return imu_data



# def get_robot_imu_data(env):
#     data = env.scene["robot"].data

#     # quaternion from root_state_w
#     root_state = data._root_state_w  # [B, 13]: [pos(3), quat(4), vel(3), ang_vel(3)]
#     quaternion = root_state[:, 3:7]

#     # gyroscope from root_state_w angular velocity
#     gyroscope = root_state[:, 10:13]

#     # accelerometer: root COM acceleration
#     accelerometer = data._root_com_acc_w[:, :3]  # [B, 3]

#     # convert quaternion to RPY (Roll, Pitch, Yaw)
#     rpy = quaternion_to_rpy(quaternion)

#     return {
#         "quaternion": quaternion,
#         "gyroscope": gyroscope,
#         "accelerometer": accelerometer,
#         "rpy": rpy
#     }

def quaternion_to_rpy(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion (w, x, y, z) to roll, pitch, yaw (radians).
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clamp(t2, -1.0, 1.0)
    pitch = torch.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(t3, t4)

    return torch.stack([roll, pitch, yaw], dim=1)