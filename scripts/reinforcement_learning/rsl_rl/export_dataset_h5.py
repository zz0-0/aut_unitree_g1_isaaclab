# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to export dataset from Isaac Lab simulation with multiple robots running in parallel.

This script collects:
1. Contact states (foot contacts)
2. Ground reaction forces (GRF)
3. Joint states (positions, velocities, torques)
4. Terrain information
5. Robot base states

Dataset is stored in HDF5 format for efficient storage and retrieval.
"""

import argparse
import h5py
import numpy as np
import os
import sys
import time
import torch
from datetime import datetime
from pathlib import Path

# Add paths
sys.path.append("/home/ryz5920/Project/IsaacLab/source")
sys.path.append("/home/ryz5920/Project/aut_unitree_g1_isaaclab/source")

# Note: Skipping AppLauncher import to avoid omni module dependency
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# Add argparse arguments
parser = argparse.ArgumentParser(
    description="Export dataset from Isaac Lab simulation."
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
    "--output_dir", type=str, default="datasets", help="Directory to save the dataset."
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

# Append AppLauncher cli args (disabled - not needed for headless operation)
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
import importlib

importlib.import_module(
    "aut_unitree_g1_isaaclab.tasks.manager_based.g1_rough_locomotion_dataset"
)
importlib.import_module(
    "aut_unitree_g1_isaaclab.tasks.manager_based.go2_rough_locomotion_dataset"
)
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper


class DatasetCollector:
    """Collects and stores simulation data in HDF5 format."""

    def __init__(
        self,
        output_dir: str,
        num_envs: int,
        episode_length: int,
        robot_type: str = "robot",
        joint_names: list[str] | None = None,
        action_names: list[str] | None = None,
        foot_body_names: list[str] | None = None,
        field_metadata: dict[str, dict[str, str]] | None = None,
        global_metadata: dict[str, str | int | float] | None = None,
    ):
        """
        Initialize the dataset collector.

        Args:
            output_dir: Directory to save the dataset
            num_envs: Number of parallel environments
            episode_length: Maximum steps per episode
            robot_type: Type of robot (e.g., 'g1', 'go2') for filename
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_envs = num_envs
        self.episode_length = episode_length
        self.robot_type = robot_type.lower()
        self.joint_names = list(joint_names) if joint_names is not None else None
        self.action_names = list(action_names) if action_names is not None else None
        self.foot_body_names = (
            list(foot_body_names) if foot_body_names is not None else None
        )
        self.field_metadata = field_metadata if field_metadata is not None else {}
        self.global_metadata = global_metadata if global_metadata is not None else {}

        # Create timestamped filename with robot type
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = (
            self.output_dir / f"{self.robot_type}_locomotion_dataset_{timestamp}.h5"
        )

        # Initialize data buffers
        self.episode_data = {env_id: [] for env_id in range(num_envs)}
        self.episode_count = {env_id: 0 for env_id in range(num_envs)}
        self._initialize_hdf5_file()

        print(f"[DatasetCollector] Initialized. Will save to: {self.filename}")
        print(
            f"[DatasetCollector] Collecting data from {num_envs} parallel {self.robot_type.upper()} robots"
        )
        if self.joint_names is not None:
            print(
                f"[DatasetCollector] Saved {len(self.joint_names)} joint names in file metadata"
            )

    @staticmethod
    def _write_string_list_dataset(group, dataset_name: str, values: list[str] | None):
        """Write a UTF-8 string list dataset once if values are provided."""
        if values is None or dataset_name in group:
            return
        str_dtype = h5py.string_dtype(encoding="utf-8")
        group.create_dataset(
            dataset_name, data=np.asarray(values, dtype=object), dtype=str_dtype
        )

    def _initialize_hdf5_file(self):
        """Create dataset file and write static metadata once."""
        with h5py.File(self.filename, "a") as f:
            metadata_group = f.require_group("metadata")

            # Global metadata
            metadata_group.attrs["schema_version"] = "1.0"
            metadata_group.attrs["robot_type"] = self.robot_type
            metadata_group.attrs["num_envs"] = self.num_envs
            metadata_group.attrs["episode_length_max"] = self.episode_length
            for key, value in self.global_metadata.items():
                metadata_group.attrs[key] = value

            # Name mappings
            if self.joint_names is not None:
                self._write_string_list_dataset(
                    metadata_group, "joint_names", self.joint_names
                )
                metadata_group.attrs["num_joints"] = len(self.joint_names)
            if self.action_names is not None:
                self._write_string_list_dataset(
                    metadata_group, "action_names", self.action_names
                )
                metadata_group.attrs["num_actions"] = len(self.action_names)
            if self.foot_body_names is not None:
                self._write_string_list_dataset(
                    metadata_group, "foot_body_names", self.foot_body_names
                )
                metadata_group.attrs["num_feet"] = len(self.foot_body_names)

            # Per-field metadata for self-describing datasets
            fields_group = metadata_group.require_group("fields")
            for field_name, field_info in self.field_metadata.items():
                field_group = fields_group.require_group(field_name)
                for attr_name, attr_value in field_info.items():
                    field_group.attrs[attr_name] = attr_value

    def collect_step_data(self, env, obs, actions):
        """Collect data for a single timestep from all environments.

        Args:
            env: Isaac Lab environment.
            obs: Observations from the environment.
            actions: Actions taken in this step.

        Returns:
            Dictionary containing collected data for this step.
        """
        scene = env.unwrapped.scene
        robot = scene["robot"]
        contact_forces = scene["contact_forces"]

        # Identify foot body indices
        foot_body_names = [
            name
            for name in contact_forces.body_names
            if "ankle_roll" in name.lower() or name.lower().endswith("foot")
        ]
        foot_indices = [
            contact_forces.body_names.index(name) for name in foot_body_names
        ]

        # Collect core data fields (RECOMMENDED for credibility)
        force_contact = (
            torch.norm(contact_forces.data.net_forces_w[:, foot_indices, :], dim=-1)
            > 1.0
        )
        if hasattr(contact_forces.data, "current_contact_time"):
            time_contact = (
                contact_forces.data.current_contact_time[:, foot_indices] > 0.0
            )
            contact_states = torch.logical_or(time_contact, force_contact)
        else:
            contact_states = force_contact

        step_data = {
            # Timestamps
            "timestamp": time.time(),
            # Center of mass data (CORE)
            "root_com_pos_w": robot.data.root_com_pos_w.cpu().numpy(),
            "root_com_quat_w": robot.data.root_com_quat_w.cpu().numpy(),
            "root_com_lin_vel_w": robot.data.root_com_lin_vel_w.cpu().numpy(),
            "root_com_ang_vel_w": robot.data.root_com_ang_vel_w.cpu().numpy(),
            # IMU sensor data (CORE)
            "imu_lin_acc_b": scene["imu"].data.lin_acc_b.cpu().numpy(),
            "imu_ang_vel_b": scene["imu"].data.ang_vel_b.cpu().numpy(),
            # Joint states (CORE)
            "joint_pos": robot.data.joint_pos.cpu().numpy(),
            "joint_vel": robot.data.joint_vel.cpu().numpy(),
            "joint_acc": robot.data.joint_acc.cpu().numpy(),  # RECOMMENDED: dynamics
            "joint_torques": robot.data.applied_torque.cpu().numpy(),
            # Actions
            "actions": actions.cpu().numpy() if torch.is_tensor(actions) else actions,
            # Contact information (CORE)
            "contact_forces": contact_forces.data.net_forces_w[:, foot_indices, :]
            .cpu()
            .numpy(),
            "contact_states": contact_states.cpu().numpy(),
            # Foot kinematics (for foot node features)
            "foot_pos_w": robot.data.body_link_pos_w[:, foot_indices, :].cpu().numpy(),
            "foot_lin_vel_w": robot.data.body_link_lin_vel_w[:, foot_indices, :]
            .cpu()
            .numpy(),
        }

        return step_data

    def add_episode_step(self, env_id: int, step_data: dict):
        """Add a step to the current episode for a specific environment."""
        self.episode_data[env_id].append(step_data)

    def finalize_episode(self, env_id: int, episode_reward: float, episode_length: int):
        """
        Finalize and save an episode for a specific environment.

        Args:
            env_id: Environment ID
            episode_reward: Total reward for the episode
            episode_length: Actual length of the episode
        """
        if len(self.episode_data[env_id]) == 0:
            return

        episode_num = self.episode_count[env_id]

        # Save episode to HDF5
        with h5py.File(self.filename, "a") as f:
            # Create group for this episode
            episode_group = f.create_group(f"env_{env_id}/episode_{episode_num}")

            # Store episode metadata
            episode_group.attrs["episode_reward"] = episode_reward
            episode_group.attrs["episode_length"] = episode_length
            episode_group.attrs["env_id"] = env_id

            # Stack all step data
            for key in self.episode_data[env_id][0].keys():
                if self.episode_data[env_id][0][key] is not None:
                    # Stack data from all steps
                    data_list = [step[key] for step in self.episode_data[env_id]]
                    if isinstance(data_list[0], np.ndarray):
                        stacked_data = np.stack(data_list, axis=0)
                        episode_group.create_dataset(
                            key, data=stacked_data, compression="gzip"
                        )
                    else:
                        episode_group.create_dataset(key, data=data_list)

        # Clear episode buffer
        self.episode_data[env_id] = []
        self.episode_count[env_id] += 1

    def get_dataset_stats(self):
        """Get statistics about the collected dataset."""
        total_episodes = sum(self.episode_count.values())
        total_steps = 0

        if self.filename.exists():
            with h5py.File(self.filename, "r") as f:
                for env_key in f.keys():
                    for episode_key in f[env_key].keys():
                        episode_length = f[env_key][episode_key].attrs["episode_length"]
                        total_steps += episode_length

        return {
            "total_episodes": total_episodes,
            "total_steps": total_steps,
            "episodes_per_env": self.episode_count,
        }


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Main function to collect dataset."""

    # Override configurations
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

    # Override terrain difficulty if specified
    if args_cli.max_terrain_level is not None:
        env_cfg.scene.terrain.max_init_terrain_level = args_cli.max_terrain_level

    # Extract robot type from task name (e.g., "G1-Rough-Locomotion-Dataset-v0" -> "g1")
    robot_type = "robot"  # default
    if "g1" in args_cli.task.lower():
        robot_type = "g1"
    elif "go2" in args_cli.task.lower():
        robot_type = "go2"
    elif "go1" in args_cli.task.lower():
        robot_type = "go1"

    print("\n" + "=" * 80)
    print("DATASET EXPORT CONFIGURATION")
    print("=" * 80)
    print(f"Task: {args_cli.task}")
    print(f"Robot type: {robot_type.upper()}")
    print(f"Number of parallel robots: {args_cli.num_envs}")
    print(f"Episodes per robot: {args_cli.num_episodes}")
    print(f"Max episode length: {args_cli.episode_length}")
    print(f"Output directory: {args_cli.output_dir}")
    print(f"Random seed: {args_cli.seed}")
    print("=" * 80 + "\n")

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load policy or use random policy
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

    # Resolve metadata names used for self-describing dataset
    robot = env.unwrapped.scene["robot"]
    contact_forces = env.unwrapped.scene["contact_forces"]
    foot_body_names = [
        name
        for name in contact_forces.body_names
        if "ankle_roll" in name.lower() or name.lower().endswith("foot")
    ]

    # Get number of actions from environment
    num_actions = env.unwrapped.action_manager.total_action_dim
    action_names = (
        list(robot.joint_names)
        if len(robot.joint_names) == num_actions
        else [f"action_{i}" for i in range(num_actions)]
    )

    field_metadata = {
        "timestamp": {
            "description": "Wall-clock timestamp when this step was collected",
            "units": "seconds",
            "shape_per_step": "[]",
        },
        "root_com_pos_w": {
            "description": "Root center-of-mass position in world frame",
            "units": "meters",
            "frame": "world",
            "shape_per_step": "[3]",
        },
        "root_com_quat_w": {
            "description": "Root center-of-mass orientation quaternion in world frame",
            "units": "unitless",
            "frame": "world",
            "shape_per_step": "[4]",
            "quaternion_order": "[w, x, y, z]",
        },
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
        "imu_lin_acc_b": {
            "description": "IMU linear acceleration in body frame",
            "units": "m/s^2",
            "frame": "body",
            "shape_per_step": "[3]",
        },
        "imu_ang_vel_b": {
            "description": "IMU angular velocity in body frame",
            "units": "rad/s",
            "frame": "body",
            "shape_per_step": "[3]",
        },
        "joint_pos": {
            "description": "Joint positions ordered by metadata/joint_names",
            "units": "radians",
            "index_map": "metadata/joint_names",
            "shape_per_step": "[num_joints]",
        },
        "joint_vel": {
            "description": "Joint velocities ordered by metadata/joint_names",
            "units": "rad/s",
            "index_map": "metadata/joint_names",
            "shape_per_step": "[num_joints]",
        },
        "joint_acc": {
            "description": "Joint accelerations ordered by metadata/joint_names",
            "units": "rad/s^2",
            "index_map": "metadata/joint_names",
            "shape_per_step": "[num_joints]",
        },
        "joint_torques": {
            "description": "Applied joint torques ordered by metadata/joint_names",
            "units": "N*m",
            "index_map": "metadata/joint_names",
            "shape_per_step": "[num_joints]",
        },
        "actions": {
            "description": "Policy actions ordered by metadata/action_names",
            "units": "unitless",
            "index_map": "metadata/action_names",
            "shape_per_step": "[num_actions]",
        },
        "contact_forces": {
            "description": "Net contact force vectors for selected feet in world frame",
            "units": "N",
            "frame": "world",
            "index_map": "metadata/foot_body_names",
            "shape_per_step": "[num_feet, 3]",
        },
        "contact_states": {
            "description": "Boolean contact states for selected feet",
            "units": "boolean",
            "index_map": "metadata/foot_body_names",
            "shape_per_step": "[num_feet]",
            "true_definition": "current_contact_time > 0 OR ||contact_forces|| > contact_force_threshold_N",
        },
        "foot_pos_w": {
            "description": "Foot link positions in world frame",
            "units": "meters",
            "frame": "world",
            "index_map": "metadata/foot_body_names",
            "shape_per_step": "[num_feet, 3]",
        },
        "foot_lin_vel_w": {
            "description": "Foot link linear velocities in world frame",
            "units": "m/s",
            "frame": "world",
            "index_map": "metadata/foot_body_names",
            "shape_per_step": "[num_feet, 3]",
        },
    }

    # Initialize dataset collector
    collector = DatasetCollector(
        args_cli.output_dir,
        args_cli.num_envs,
        args_cli.episode_length,
        robot_type,
        joint_names=robot.joint_names,
        action_names=action_names,
        foot_body_names=foot_body_names,
        field_metadata=field_metadata,
        global_metadata={
            "task": args_cli.task,
            "seed": args_cli.seed,
            "contact_force_threshold_N": 1.0,
            "contact_time_threshold_s": 0.0,
        },
    )

    # Reset environment
    obs = env.get_observations()
    episode_rewards = torch.zeros(args_cli.num_envs, device=env.unwrapped.device)
    episode_lengths = torch.zeros(
        args_cli.num_envs, dtype=torch.int32, device=env.unwrapped.device
    )

    # Track progress
    total_episodes_target = args_cli.num_envs * args_cli.num_episodes
    completed_episodes = 0

    # Track which environments have completed all episodes
    envs_completed = torch.zeros(
        args_cli.num_envs, dtype=torch.bool, device=env.unwrapped.device
    )

    print("\n[INFO] Starting data collection...")
    print(
        f"[INFO] Target: {args_cli.num_episodes} episodes per robot, {total_episodes_target} total episodes"
    )
    print(f"[INFO] Will automatically stop when all robots complete their episodes\n")
    start_time = time.time()

    # Main collection loop
    while not envs_completed.all():
        with torch.inference_mode():
            # Get actions
            if policy is not None:
                actions = policy(obs)
            else:
                # Random actions
                actions = torch.randn(
                    args_cli.num_envs, num_actions, device=env.unwrapped.device
                )
                actions = torch.clamp(actions, -1.0, 1.0)

            # Step environment
            obs, rewards, dones, extras = env.step(actions)

            # Collect step data for all environments
            step_data = collector.collect_step_data(env.unwrapped, obs, actions)

            # Add to episode buffers
            for env_id in range(args_cli.num_envs):
                # Extract data for this specific environment
                env_step_data = {}
                for key, value in step_data.items():
                    if value is not None:
                        # Check if it's an array with environment dimension
                        if (
                            isinstance(value, np.ndarray)
                            and len(value.shape) > 0
                            and value.shape[0] == args_cli.num_envs
                        ):
                            env_step_data[key] = value[env_id]
                        else:
                            # Scalar or global value (like timestamp)
                            env_step_data[key] = value
                    else:
                        env_step_data[key] = value

                collector.add_episode_step(env_id, env_step_data)
                episode_rewards[env_id] += rewards[env_id]
                episode_lengths[env_id] += 1

            # Check for episode completion
            for env_id in range(args_cli.num_envs):
                # Skip if this environment already completed all episodes
                if envs_completed[env_id]:
                    continue

                if dones[env_id] or episode_lengths[env_id] >= args_cli.episode_length:
                    # Finalize episode
                    if collector.episode_count[env_id] < args_cli.num_episodes:
                        collector.finalize_episode(
                            env_id,
                            episode_rewards[env_id].item(),
                            episode_lengths[env_id].item(),
                        )
                        completed_episodes += 1

                        # Check if this env completed all its episodes
                        if collector.episode_count[env_id] >= args_cli.num_episodes:
                            envs_completed[env_id] = True
                            print(
                                f"[Robot {env_id}] Completed all {args_cli.num_episodes} episodes!"
                            )

                        # Progress update
                        if completed_episodes % 10 == 0:
                            elapsed_time = time.time() - start_time
                            progress = 100 * completed_episodes / total_episodes_target
                            remaining_envs = (~envs_completed).sum().item()
                            print(
                                f"[Progress] {completed_episodes}/{total_episodes_target} episodes "
                                f"({progress:.1f}%) | {remaining_envs} robots still collecting | Time: {elapsed_time:.1f}s"
                            )

                    # Reset episode tracking
                    episode_rewards[env_id] = 0
                    episode_lengths[env_id] = 0

    # Final statistics
    elapsed_time = time.time() - start_time
    stats = collector.get_dataset_stats()

    print("\n" + "=" * 80)
    print("DATA COLLECTION COMPLETE")
    print("=" * 80)
    print(f"Total episodes collected: {stats['total_episodes']}")
    print(f"Total steps collected: {stats['total_steps']}")
    print(f"Collection time: {elapsed_time:.1f}s")
    print(f"Dataset saved to: {collector.filename}")
    print("=" * 80 + "\n")

    # Close environment
    env.close()


if __name__ == "__main__":
    main()
    # simulation_app was never initialized, so don't close it
