# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to export dataset from Isaac Lab simulation into NumPy memmap files.

This format is optimized for fast random/sequential slicing from Torch pipelines,
while preserving self-describing metadata in a companion metadata.json file.
"""

import argparse
import importlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap

# Add paths
sys.path.append("/home/ryz5920/Project/IsaacLab/source")
sys.path.append("/home/ryz5920/Project/aut_unitree_g1_isaaclab/source")

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# Add argparse arguments
parser = argparse.ArgumentParser(
    description="Export dataset from Isaac Lab simulation into NumPy memmap format."
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=512,
    help="Number of parallel environments (robots).",
)
parser.add_argument(
    "--task",
    type=str,
    default="G1-Rough-Locomotion-Dataset-v0",
    help="Name of the task.",
)
parser.add_argument(
    "--num_episodes",
    type=int,
    default=100,
    help="Number of episodes to collect per environment.",
)
parser.add_argument(
    "--episode_length", type=int, default=1000, help="Maximum steps per episode."
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="datasets_memmap",
    help="Directory to save the memmap dataset folder.",
)
parser.add_argument(
    "--seed", type=int, default=42, help="Random seed for reproducibility."
)
parser.add_argument(
    "--policy_checkpoint",
    type=str,
    default=None,
    help="Path to policy checkpoint for data collection.",
)
parser.add_argument(
    "--use_random_policy",
    action="store_true",
    help="Use random policy instead of trained policy.",
)
parser.add_argument(
    "--max_terrain_level",
    type=int,
    default=None,
    help="Maximum terrain difficulty level (0-5). If None, uses default curriculum.",
)
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)

# Add RSL-RL specific arguments that cli_args.update_rsl_rl_cfg expects
cli_args.add_rsl_rl_args(parser)

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# Parse arguments
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

import aut_unitree_g1_isaaclab  # noqa: F401

importlib.import_module(
    "aut_unitree_g1_isaaclab.tasks.manager_based.g1_rough_locomotion_dataset"
)
importlib.import_module(
    "aut_unitree_g1_isaaclab.tasks.manager_based.go2_rough_locomotion_dataset"
)
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper


def _to_numpy_dtype_name(dtype: np.dtype) -> str:
    return np.dtype(dtype).name


def infer_robot_type(task_name: str) -> str:
    robot_type = "robot"
    task_lower = task_name.lower()
    if "g1" in task_lower:
        robot_type = "g1"
    elif "go2" in task_lower:
        robot_type = "go2"
    elif "go1" in task_lower:
        robot_type = "go1"
    return robot_type


class MemmapDatasetCollector:
    """Collects simulation data into per-field NumPy memmap arrays."""

    def __init__(
        self,
        output_dir: str,
        robot_type: str,
        num_envs: int,
        num_episodes: int,
        episode_length: int,
        field_specs: dict[str, dict],
        joint_names: list[str],
        foot_body_names: list[str],
        hand_body_names: list[str],
        field_metadata: dict[str, dict[str, str]],
        global_metadata: dict[str, str | int | float],
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.robot_type = robot_type
        self.num_envs = num_envs
        self.num_episodes = num_episodes
        self.episode_length = episode_length

        self.field_specs = field_specs
        self.field_metadata = field_metadata
        self.global_metadata = global_metadata

        self.joint_names = list(joint_names)
        self.foot_body_names = list(foot_body_names)
        self.hand_body_names = list(hand_body_names)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dataset_dir = (
            self.output_dir / f"{self.robot_type}_locomotion_memmap_{timestamp}"
        )
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        self.data_memmaps: dict[str, np.memmap] = {}
        self.episode_reward = open_memmap(
            self.dataset_dir / "episode_reward.npy",
            mode="w+",
            dtype=np.float32,
            shape=(num_envs, num_episodes),
        )
        self.episode_length_actual = open_memmap(
            self.dataset_dir / "episode_length_actual.npy",
            mode="w+",
            dtype=np.int32,
            shape=(num_envs, num_episodes),
        )
        self.episode_done = open_memmap(
            self.dataset_dir / "episode_done.npy",
            mode="w+",
            dtype=np.bool_,
            shape=(num_envs, num_episodes),
        )
        self.valid_steps = open_memmap(
            self.dataset_dir / "valid_steps.npy",
            mode="w+",
            dtype=np.bool_,
            shape=(num_envs, num_episodes, episode_length),
        )

        self.episode_reward.fill(0)
        self.episode_length_actual.fill(0)
        self.episode_done.fill(False)
        self.valid_steps.fill(False)

        for field_name, spec in self.field_specs.items():
            dtype = np.dtype(spec["dtype"])
            field_shape = tuple(spec["shape_per_step"])
            full_shape = (num_envs, num_episodes, episode_length, *field_shape)
            self.data_memmaps[field_name] = open_memmap(
                self.dataset_dir / f"{field_name}.npy",
                mode="w+",
                dtype=dtype,
                shape=full_shape,
            )

        self.current_episode_idx = np.zeros(num_envs, dtype=np.int32)
        self.current_step_idx = np.zeros(num_envs, dtype=np.int32)

        self._write_metadata_json()

        print(f"[MemmapCollector] Initialized. Saving to folder: {self.dataset_dir}")
        print(f"[MemmapCollector] Data fields: {len(self.data_memmaps)}")

    def _write_metadata_json(self):
        field_files = {name: f"{name}.npy" for name in self.field_specs.keys()}
        field_shapes = {
            name: [
                self.num_envs,
                self.num_episodes,
                self.episode_length,
                *list(self.field_specs[name]["shape_per_step"]),
            ]
            for name in self.field_specs
        }
        field_dtypes = {
            name: _to_numpy_dtype_name(self.field_specs[name]["dtype"])
            for name in self.field_specs
        }

        metadata = {
            "format": "numpy_memmap",
            "schema_version": "1.0",
            "robot_type": self.robot_type,
            "num_envs": self.num_envs,
            "num_episodes": self.num_episodes,
            "episode_length_max": self.episode_length,
            "global_metadata": self.global_metadata,
            "index_maps": {
                "joint_names": self.joint_names,
                "foot_body_names": self.foot_body_names,
                "hand_body_names": self.hand_body_names,
            },
            "files": {
                "fields": field_files,
                "episode_reward": "episode_reward.npy",
                "episode_length_actual": "episode_length_actual.npy",
                "episode_done": "episode_done.npy",
                "valid_steps": "valid_steps.npy",
            },
            "field_shapes": field_shapes,
            "field_dtypes": field_dtypes,
            "field_metadata": self.field_metadata,
        }

        with open(self.dataset_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def add_step(self, env_id: int, step_data: dict):
        episode_idx = int(self.current_episode_idx[env_id])
        if episode_idx >= self.num_episodes:
            return

        step_idx = int(self.current_step_idx[env_id])
        if step_idx >= self.episode_length:
            return

        for key, value in step_data.items():
            if key in self.data_memmaps and value is not None:
                self.data_memmaps[key][env_id, episode_idx, step_idx] = value

        self.valid_steps[env_id, episode_idx, step_idx] = True
        self.current_step_idx[env_id] += 1

    def finalize_episode(self, env_id: int, episode_reward: float):
        episode_idx = int(self.current_episode_idx[env_id])
        if episode_idx >= self.num_episodes:
            return

        length = int(min(self.current_step_idx[env_id], self.episode_length))
        self.episode_reward[env_id, episode_idx] = episode_reward
        self.episode_length_actual[env_id, episode_idx] = length
        self.episode_done[env_id, episode_idx] = True

        self.current_episode_idx[env_id] += 1
        self.current_step_idx[env_id] = 0

    def flush(self):
        for arr in self.data_memmaps.values():
            arr.flush()
        self.episode_reward.flush()
        self.episode_length_actual.flush()
        self.episode_done.flush()
        self.valid_steps.flush()

    def get_dataset_stats(self):
        completed_mask = np.asarray(self.episode_done)
        total_episodes = int(completed_mask.sum())
        total_steps = int(np.asarray(self.episode_length_actual).sum())
        episodes_per_env = (
            np.asarray(self.current_episode_idx).astype(np.int32).tolist()
        )
        return {
            "total_episodes": total_episodes,
            "total_steps": total_steps,
            "episodes_per_env": episodes_per_env,
        }


def collect_step_data(env, actions, foot_indices: list[int], hand_indices: list[int]):
    """Collect batched step data for all environments."""
    scene = env.unwrapped.scene
    robot = scene["robot"]
    contact_forces = scene["contact_forces"]

    force_contact = (
        torch.norm(contact_forces.data.net_forces_w[:, foot_indices, :], dim=-1) > 1.0
    )
    if hasattr(contact_forces.data, "current_contact_time"):
        time_contact = contact_forces.data.current_contact_time[:, foot_indices] > 0.0
        contact_states = torch.logical_or(time_contact, force_contact)
    else:
        contact_states = force_contact

    return {
        "root_com_lin_vel_w": robot.data.root_com_lin_vel_w.cpu().numpy(),
        "root_com_ang_vel_w": robot.data.root_com_ang_vel_w.cpu().numpy(),
        "imu_lin_acc": scene["imu"].data.lin_acc_b.cpu().numpy(),
        "imu_ang_vel": scene["imu"].data.ang_vel_b.cpu().numpy(),
        "joint_pos": robot.data.joint_pos.cpu().numpy(),
        "joint_vel": robot.data.joint_vel.cpu().numpy(),
        "joint_torque": robot.data.applied_torque.cpu().numpy(),
        "contact_forces": contact_forces.data.net_forces_w[:, foot_indices, :]
        .cpu()
        .numpy(),
        "contact_states": contact_states.cpu().numpy(),
        "foot_pos_w": robot.data.body_link_pos_w[:, foot_indices, :].cpu().numpy(),
        "foot_lin_vel_w": robot.data.body_link_lin_vel_w[:, foot_indices, :]
        .cpu()
        .numpy(),
        "hand_pos_w": (
            robot.data.body_link_pos_w[:, hand_indices, :].cpu().numpy()
            if len(hand_indices) > 0
            else None
        ),
        "hand_lin_vel_w": (
            robot.data.body_link_lin_vel_w[:, hand_indices, :].cpu().numpy()
            if len(hand_indices) > 0
            else None
        ),
    }


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

    if args_cli.max_terrain_level is not None:
        env_cfg.scene.terrain.max_init_terrain_level = args_cli.max_terrain_level

    robot_type = infer_robot_type(args_cli.task)

    print("\n" + "=" * 80)
    print("MEMMAP DATASET EXPORT CONFIGURATION")
    print("=" * 80)
    print(f"Task: {args_cli.task}")
    print(f"Robot type: {robot_type.upper()}")
    print(f"Number of parallel robots: {args_cli.num_envs}")
    print(f"Episodes per robot: {args_cli.num_episodes}")
    print(f"Max episode length: {args_cli.episode_length}")
    print(f"Output directory: {args_cli.output_dir}")
    print(f"Random seed: {args_cli.seed}")
    print("=" * 80 + "\n")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    policy = None
    if not args_cli.use_random_policy and args_cli.policy_checkpoint:
        print(f"[INFO] Loading policy from: {args_cli.policy_checkpoint}")
        resume_path = retrieve_file_path(args_cli.policy_checkpoint)
        runner = OnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
    else:
        print("[INFO] Using random policy for data collection")

    robot = env.unwrapped.scene["robot"]
    contact_forces = env.unwrapped.scene["contact_forces"]

    foot_body_names = [
        name
        for name in contact_forces.body_names
        if "ankle_roll" in name.lower() or name.lower().endswith("foot")
    ]
    foot_indices = [contact_forces.body_names.index(name) for name in foot_body_names]

    # Detect hand (wrist-yaw / end-effector) bodies from the robot links.
    # Leaf wrist joints in BHMG are the terminal arm joints; their child links
    # are the hand nodes. Match by "wrist_yaw" to get only the distal wrist link.
    hand_body_names = [
        name
        for name in robot.body_names
        if "wrist_yaw" in name.lower() or name.lower().endswith("hand")
    ]
    hand_indices = [list(robot.body_names).index(name) for name in hand_body_names]

    joint_names = list(robot.joint_names)
    num_actions = env.unwrapped.action_manager.total_action_dim

    field_specs = {
        "root_com_lin_vel_w": {"dtype": np.float32, "shape_per_step": (3,)},
        "root_com_ang_vel_w": {"dtype": np.float32, "shape_per_step": (3,)},
        "imu_lin_acc": {"dtype": np.float32, "shape_per_step": (3,)},
        "imu_ang_vel": {"dtype": np.float32, "shape_per_step": (3,)},
        "joint_pos": {"dtype": np.float32, "shape_per_step": (len(joint_names),)},
        "joint_vel": {"dtype": np.float32, "shape_per_step": (len(joint_names),)},
        "joint_torque": {"dtype": np.float32, "shape_per_step": (len(joint_names),)},
        "contact_forces": {
            "dtype": np.float32,
            "shape_per_step": (len(foot_body_names), 3),
        },
        "contact_states": {
            "dtype": np.bool_,
            "shape_per_step": (len(foot_body_names),),
        },
        "foot_pos_w": {
            "dtype": np.float32,
            "shape_per_step": (len(foot_body_names), 3),
        },
        "foot_lin_vel_w": {
            "dtype": np.float32,
            "shape_per_step": (len(foot_body_names), 3),
        },
    }
    if len(hand_body_names) > 0:
        field_specs["hand_pos_w"] = {
            "dtype": np.float32,
            "shape_per_step": (len(hand_body_names), 3),
        }
        field_specs["hand_lin_vel_w"] = {
            "dtype": np.float32,
            "shape_per_step": (len(hand_body_names), 3),
        }

    field_metadata = {
        "root_com_lin_vel_w": {
            "description": "Root center-of-mass linear velocity in world frame",
            "units": "m/s",
            "frame": "world",
            "shape_per_step": "[3]",
        },
        "root_com_ang_vel_w": {
            "description": "Root center-of-mass angular velocity in world frame",
            "units": "rad/s",
            "frame": "world",
            "shape_per_step": "[3]",
        },
        "imu_lin_acc": {
            "description": "IMU linear acceleration in body frame",
            "units": "m/s^2",
            "frame": "body",
            "shape_per_step": "[3]",
        },
        "imu_ang_vel": {
            "description": "IMU angular velocity in body frame",
            "units": "rad/s",
            "frame": "body",
            "shape_per_step": "[3]",
        },
        "joint_pos": {
            "description": "Joint positions ordered by index_maps/joint_names",
            "units": "radians",
            "index_map": "index_maps/joint_names",
            "shape_per_step": "[num_joints]",
        },
        "joint_vel": {
            "description": "Joint velocities ordered by index_maps/joint_names",
            "units": "rad/s",
            "index_map": "index_maps/joint_names",
            "shape_per_step": "[num_joints]",
        },
        "joint_torque": {
            "description": "Applied joint torques ordered by index_maps/joint_names",
            "units": "N*m",
            "index_map": "index_maps/joint_names",
            "shape_per_step": "[num_joints]",
        },
        "contact_forces": {
            "description": "Net contact force vectors for selected feet in world frame",
            "units": "N",
            "frame": "world",
            "index_map": "index_maps/foot_body_names",
            "shape_per_step": "[num_feet, 3]",
        },
        "contact_states": {
            "description": "Boolean contact states for selected feet",
            "units": "boolean",
            "index_map": "index_maps/foot_body_names",
            "shape_per_step": "[num_feet]",
            "true_definition": "current_contact_time > 0 OR ||contact_forces|| > contact_force_threshold_N",
        },
        "foot_pos_w": {
            "description": "Foot link positions in world frame",
            "units": "meters",
            "frame": "world",
            "index_map": "index_maps/foot_body_names",
            "shape_per_step": "[num_feet, 3]",
        },
        "foot_lin_vel_w": {
            "description": "Foot link linear velocities in world frame",
            "units": "m/s",
            "frame": "world",
            "index_map": "index_maps/foot_body_names",
            "shape_per_step": "[num_feet, 3]",
        },
    }
    if len(hand_body_names) > 0:
        field_metadata["hand_pos_w"] = {
            "description": "Hand link positions in world frame",
            "units": "meters",
            "frame": "world",
            "index_map": "index_maps/hand_body_names",
            "shape_per_step": "[num_hands, 3]",
        }
        field_metadata["hand_lin_vel_w"] = {
            "description": "Hand link linear velocities in world frame",
            "units": "m/s",
            "frame": "world",
            "index_map": "index_maps/hand_body_names",
            "shape_per_step": "[num_hands, 3]",
        }

    collector = MemmapDatasetCollector(
        output_dir=args_cli.output_dir,
        robot_type=robot_type,
        num_envs=args_cli.num_envs,
        num_episodes=args_cli.num_episodes,
        episode_length=args_cli.episode_length,
        field_specs=field_specs,
        joint_names=joint_names,
        foot_body_names=foot_body_names,
        hand_body_names=hand_body_names,
        field_metadata=field_metadata,
        global_metadata={
            "task": args_cli.task,
            "seed": args_cli.seed,
            "contact_force_threshold_N": 1.0,
            "contact_time_threshold_s": 0.0,
        },
    )

    obs = env.get_observations()
    episode_rewards = torch.zeros(args_cli.num_envs, device=env.unwrapped.device)

    total_episodes_target = args_cli.num_envs * args_cli.num_episodes
    completed_episodes = 0
    envs_completed = torch.zeros(
        args_cli.num_envs, dtype=torch.bool, device=env.unwrapped.device
    )

    print("\n[INFO] Starting memmap data collection...")
    print(
        f"[INFO] Target: {args_cli.num_episodes} episodes per robot, {total_episodes_target} total episodes"
    )
    start_time = time.time()

    while not envs_completed.all():
        with torch.inference_mode():
            if policy is not None:
                actions = policy(obs)
            else:
                actions = torch.randn(
                    args_cli.num_envs, num_actions, device=env.unwrapped.device
                )
                actions = torch.clamp(actions, -1.0, 1.0)

            obs, rewards, dones, extras = env.step(actions)
            step_data = collect_step_data(
                env.unwrapped, actions, foot_indices, hand_indices
            )

            for env_id in range(args_cli.num_envs):
                if envs_completed[env_id]:
                    continue

                env_step_data = {}
                for key, value in step_data.items():
                    if value is None:
                        env_step_data[key] = value
                    elif (
                        isinstance(value, np.ndarray)
                        and len(value.shape) > 0
                        and value.shape[0] == args_cli.num_envs
                    ):
                        env_step_data[key] = value[env_id]
                    else:
                        env_step_data[key] = value

                collector.add_step(env_id, env_step_data)
                episode_rewards[env_id] += rewards[env_id]

                episode_step_count = collector.current_step_idx[env_id]
                max_length_reached = episode_step_count >= args_cli.episode_length

                if dones[env_id] or max_length_reached:
                    collector.finalize_episode(env_id, episode_rewards[env_id].item())
                    completed_episodes += 1

                    if collector.current_episode_idx[env_id] >= args_cli.num_episodes:
                        envs_completed[env_id] = True
                        print(
                            f"[Robot {env_id}] Completed all {args_cli.num_episodes} episodes!"
                        )

                    episode_rewards[env_id] = 0.0

                    if completed_episodes % 10 == 0:
                        elapsed_time = time.time() - start_time
                        progress = 100 * completed_episodes / total_episodes_target
                        remaining_envs = (~envs_completed).sum().item()
                        print(
                            f"[Progress] {completed_episodes}/{total_episodes_target} episodes "
                            f"({progress:.1f}%) | {remaining_envs} robots still collecting | Time: {elapsed_time:.1f}s"
                        )

    collector.flush()
    elapsed_time = time.time() - start_time
    stats = collector.get_dataset_stats()

    print("\n" + "=" * 80)
    print("MEMMAP DATA COLLECTION COMPLETE")
    print("=" * 80)
    print(f"Total episodes collected: {stats['total_episodes']}")
    print(f"Total steps collected: {stats['total_steps']}")
    print(f"Collection time: {elapsed_time:.1f}s")
    print(f"Dataset folder: {collector.dataset_dir}")
    print("=" * 80 + "\n")

    env.close()


if __name__ == "__main__":
    main()
