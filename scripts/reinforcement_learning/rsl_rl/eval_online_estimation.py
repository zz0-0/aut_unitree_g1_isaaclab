"""Sim-online validation: run a locomotion policy and the streaming GNN estimator
concurrently at the 50 Hz control rate and compare online estimates against
simulator ground truth, while measuring end-to-end estimation latency.

The estimator consumes only proprioception (same fields as the offline dataset)
and maintains its own causal history buffer. Latency is measured batch=1 with
CUDA synchronization, i.e. the deployment-relevant number.
"""

import argparse
import importlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[3] / "source"))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Sim-online estimation validation.")
parser.add_argument("--task", type=str, default="G1-Rough-Locomotion-Dataset-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=1500)
parser.add_argument("--policy_checkpoint", type=str, default=None)
parser.add_argument("--external_policy_onnx", type=str, default=None)
parser.add_argument("--external_action_scale", type=float, default=0.25)
parser.add_argument("--use_random_policy", action="store_true")
parser.add_argument("--max_terrain_level", type=int, default=None)
parser.add_argument("--joint_friction_mu_range", nargs=2, type=float, default=[0.0, 0.0])
parser.add_argument(
    "--joint_friction_viscous_range", nargs=2, type=float, default=[0.0, 0.0]
)
parser.add_argument("--gnn_repo", type=str, default="/home/ryz5920/Github Project/aut_om_hgnn")
parser.add_argument("--gnn_config_path", type=str, required=True)
parser.add_argument("--gnn_checkpoint_path", type=str, required=True)
parser.add_argument("--gnn_device", type=str, default="cuda")
parser.add_argument("--latency_warmup", type=int, default=50)
parser.add_argument("--output", type=str, default="/tmp/opencode/online_estimation.json")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.math import quat_apply_inverse
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import aut_unitree_g1_isaaclab  # noqa: F401
from aut_unitree_g1_isaaclab.policies import ActorMLPPolicy, UnitreeRLLabOnnxPolicy

importlib.import_module(
    "aut_unitree_g1_isaaclab.tasks.manager_based.g1_rough_locomotion_dataset"
)
importlib.import_module(
    "aut_unitree_g1_isaaclab.tasks.manager_based.go2_rough_locomotion_dataset"
)
from isaaclab_tasks.utils.hydra import hydra_task_config

sys.path.insert(0, args_cli.gnn_repo)
from src.inference import StreamingEstimator  # noqa: E402
from src.config.train_config import TrainConfig  # noqa: E402
from src.config.train_enum import OutputType  # noqa: E402

GRAVITY_VEC_W = (0.0, 0.0, -1.0)


def collect_joint_friction(robot) -> np.ndarray:
    joint_friction = torch.zeros_like(robot.data.joint_pos)
    for actuator in robot.actuators.values():
        friction_torque = getattr(actuator, "friction_torque", None)
        if friction_torque is not None and getattr(actuator, "friction_enabled", False):
            joint_friction[:, actuator.joint_indices] = friction_torque
    return joint_friction.cpu().numpy()


def build_online_data(
    unwrapped,
    foot_indices: list[int],
    hand_indices: list[int],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return (estimator frame, ground truth) for env 0 in base-frame convention."""
    scene = unwrapped.scene
    robot = scene["robot"]
    contact_forces = scene["contact_forces"]

    root_quat = robot.data.root_quat_w
    root_link_pos = robot.data.root_link_pos_w
    root_com_lin_vel = robot.data.root_com_lin_vel_w

    def to_base(vec_world: torch.Tensor) -> torch.Tensor:
        quat = root_quat.unsqueeze(1).repeat(1, vec_world.shape[1], 1)
        return quat_apply_inverse(quat, vec_world)

    foot_pos_w = robot.data.body_link_pos_w[:, foot_indices, :]
    foot_lin_vel_w = robot.data.body_link_lin_vel_w[:, foot_indices, :]
    foot_pos_b = quat_apply_inverse(
        root_quat.unsqueeze(1), foot_pos_w - root_link_pos.unsqueeze(1)
    )
    foot_lin_vel_b = quat_apply_inverse(
        root_quat.unsqueeze(1), foot_lin_vel_w - root_com_lin_vel.unsqueeze(1)
    )
    net_forces_w = contact_forces.data.net_forces_w[:, foot_indices, :]
    net_forces_b = to_base(net_forces_w)

    force_contact = torch.norm(net_forces_w, dim=-1) > 1.0
    if hasattr(contact_forces.data, "current_contact_time"):
        time_contact = contact_forces.data.current_contact_time[:, foot_indices] > 0.0
        contact_states = torch.logical_or(time_contact, force_contact)
    else:
        contact_states = force_contact

    frame = {
        "joint_pos": robot.data.joint_pos[0].cpu().numpy(),
        "joint_vel": robot.data.joint_vel[0].cpu().numpy(),
        "joint_torque": robot.data.applied_torque[0].cpu().numpy(),
        "imu_lin_acc": scene["imu"].data.lin_acc_b[0].cpu().numpy(),
        "imu_ang_vel": scene["imu"].data.ang_vel_b[0].cpu().numpy(),
        "foot_pos": foot_pos_b[0].cpu().numpy(),
        "foot_lin_vel": foot_lin_vel_b[0].cpu().numpy(),
    }
    if len(hand_indices) > 0:
        hand_pos_w = robot.data.body_link_pos_w[:, hand_indices, :]
        hand_lin_vel_w = robot.data.body_link_lin_vel_w[:, hand_indices, :]
        frame["hand_pos"] = quat_apply_inverse(
            root_quat.unsqueeze(1), hand_pos_w - root_link_pos.unsqueeze(1)
        )[0].cpu().numpy()
        frame["hand_lin_vel"] = quat_apply_inverse(
            root_quat.unsqueeze(1), hand_lin_vel_w - root_com_lin_vel.unsqueeze(1)
        )[0].cpu().numpy()

    ground_truth = {
        "CONTACT": contact_states[0].cpu().numpy().astype(np.float32),
        "GROUND_REACTION_FORCE": net_forces_b[0].cpu().numpy().reshape(-1),
        "BASE_VELOCITY": np.concatenate(
            [
                robot.data.root_com_lin_vel_b[0].cpu().numpy(),
                robot.data.root_com_ang_vel_b[0].cpu().numpy(),
            ]
        ),
        "TOTAL_GROUND_REACTION_FORCE": net_forces_b[0].sum(dim=0).cpu().numpy(),
        "BASE_ANGULAR_ACCELERATION": scene["imu"].data.ang_acc_b[0].cpu().numpy(),
        "JOINT_ACCELERATION": robot.data.joint_acc[0].cpu().numpy(),
        "JOINT_FRICTION": collect_joint_friction(robot)[0],
    }
    return frame, ground_truth


def reorder_joint_gt(values: np.ndarray, estimator: StreamingEstimator) -> np.ndarray:
    indices = estimator.feature_extractor.joint_indices_for_type(
        {"joint_names": estimator.joint_names}, "joint"
    )
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    out = np.zeros(len(indices), dtype=np.float32)
    valid = [
        (position, int(index))
        for position, index in enumerate(indices)
        if 0 <= int(index) < values.shape[0]
    ]
    if valid:
        positions = np.asarray([item[0] for item in valid], dtype=np.int64)
        raw_indices = np.asarray([item[1] for item in valid], dtype=np.int64)
        out[positions] = values[raw_indices]
    return out


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )
    if args_cli.max_terrain_level is not None:
        env_cfg.scene.terrain.max_init_terrain_level = args_cli.max_terrain_level

    friction_requested = any(
        value > 0.0
        for value in (
            *args_cli.joint_friction_mu_range,
            *args_cli.joint_friction_viscous_range,
        )
    )
    if friction_requested:
        for actuator_cfg in env_cfg.scene.robot.actuators.values():
            if hasattr(actuator_cfg, "friction_mu_range"):
                actuator_cfg.friction_mu_range = tuple(args_cli.joint_friction_mu_range)
                actuator_cfg.friction_viscous_range = tuple(
                    args_cli.joint_friction_viscous_range
                )

    if args_cli.num_envs != 1:
        raise SystemExit("Online validation currently supports --num_envs 1.")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    unwrapped = env.unwrapped

    device = torch.device(unwrapped.device)
    robot = unwrapped.scene["robot"]
    contact_forces = unwrapped.scene["contact_forces"]
    foot_body_names = [
        name
        for name in contact_forces.body_names
        if "ankle_roll" in name.lower() or name.lower().endswith("foot")
    ]
    foot_indices = [contact_forces.body_names.index(name) for name in foot_body_names]
    hand_body_names = [
        name
        for name in robot.body_names
        if "wrist_yaw" in name.lower() or name.lower().endswith("hand")
    ]
    hand_indices = [list(robot.body_names).index(name) for name in hand_body_names]

    policy = None
    if args_cli.external_policy_onnx:
        policy = UnitreeRLLabOnnxPolicy(
            args_cli.external_policy_onnx,
            unwrapped,
            action_scale=args_cli.external_action_scale,
        )
    elif args_cli.policy_checkpoint:
        checkpoint = torch.load(
            args_cli.policy_checkpoint, map_location="cpu", weights_only=False
        )
        state_dict = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint
        )
        policy = ActorMLPPolicy(state_dict, device=str(unwrapped.device))
    elif not args_cli.use_random_policy:
        raise SystemExit("Provide a policy checkpoint or --use_random_policy.")

    estimator = StreamingEstimator(
        config_path=args_cli.gnn_config_path,
        checkpoint_path=args_cli.gnn_checkpoint_path,
        joint_names=list(robot.joint_names),
        device=args_cli.gnn_device,
        repo_root=args_cli.gnn_repo,
    )
    config = TrainConfig.build_from(args_cli.gnn_config_path)
    output_types = (
        list(config.output_types) if config.output_types else [config.output_type]
    )
    num_actions = unwrapped.action_manager.total_action_dim

    sums_abs: dict[str, float] = defaultdict(float)
    sums_sq: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    contact_hits = 0
    contact_total = 0

    latencies_ms: list[float] = []
    frame_times_ms: list[float] = []
    obs = env.get_observations()
    done = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=unwrapped.device)
    history_warmup = max(0, int(getattr(estimator, "history_length", 1)) - 1)
    metrics_start = max(args_cli.latency_warmup, history_warmup)
    reset_count = 0

    with torch.inference_mode():
        for step in range(args_cli.steps):
            frame_start = time.perf_counter()
            frame, ground_truth = build_online_data(
                unwrapped, foot_indices, hand_indices
            )
            frame_time_ms = (time.perf_counter() - frame_start) * 1000.0

            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_start = time.perf_counter()
            predictions = estimator.step_all(frame)
            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_ms = (time.perf_counter() - infer_start) * 1000.0

            if step >= args_cli.latency_warmup:
                latencies_ms.append(infer_ms)
                frame_times_ms.append(frame_time_ms)

            if step >= metrics_start:
                for output_type in output_types:
                    key = output_type.value
                    if key not in predictions:
                        continue
                    pred = np.asarray(predictions[key], dtype=np.float32).reshape(-1)
                    target = np.asarray(ground_truth[key], dtype=np.float32).reshape(-1)
                    if output_type in (
                        OutputType.JOINT_ACCELERATION,
                        OutputType.JOINT_FRICTION,
                    ):
                        target = reorder_joint_gt(target, estimator)
                    if pred.shape != target.shape:
                        continue
                    sums_abs[key] += float(np.abs(pred - target).sum())
                    sums_sq[key] += float(((pred - target) ** 2).sum())
                    counts[key] += int(pred.size)
                    if output_type == OutputType.CONTACT:
                        contact_hits += int(
                            ((pred > 0.5) == (target > 0.5)).sum()
                        )
                        contact_total += int(pred.size)

            if policy is not None:
                policy_obs = (
                    obs["policy"]
                    if hasattr(obs, "keys") and "policy" in obs.keys()
                    else obs
                )
                actions = policy(policy_obs)
            else:
                actions = torch.clamp(
                    torch.randn(args_cli.num_envs, num_actions, device=unwrapped.device),
                    -1.0,
                    1.0,
                )

            obs, _, dones, _ = env.step(actions)
            done = dones
            if bool(torch.as_tensor(done).any()):
                estimator.reset()
                reset_count += 1
                metrics_start = max(metrics_start, step + 1 + history_warmup)

    metrics: dict[str, dict[str, float]] = {}
    for key, count in counts.items():
        metrics[key] = {
            "mae": sums_abs[key] / max(count, 1),
            "rmse": (sums_sq[key] / max(count, 1)) ** 0.5,
        }
    if contact_total > 0:
        metrics["CONTACT"]["accuracy"] = contact_hits / contact_total

    latency = np.asarray(latencies_ms, dtype=np.float64)
    frame_time = np.asarray(frame_times_ms, dtype=np.float64)
    result = {
        "task": args_cli.task,
        "gnn_config": args_cli.gnn_config_path,
        "gnn_checkpoint": args_cli.gnn_checkpoint_path,
        "policy": (
            args_cli.external_policy_onnx or args_cli.policy_checkpoint or "random"
        ),
        "steps": args_cli.steps,
        "num_envs": args_cli.num_envs,
        "control_dt_s": float(unwrapped.step_dt),
        "metrics_warmup_steps": history_warmup,
        "episode_resets": reset_count,
        "metrics": metrics,
        "latency_ms": {
            "infer_mean": float(latency.mean()) if latency.size else None,
            "infer_p50": float(np.percentile(latency, 50)) if latency.size else None,
            "infer_p95": float(np.percentile(latency, 95)) if latency.size else None,
            "infer_max": float(latency.max()) if latency.size else None,
            "frame_mean": float(frame_time.mean()) if frame_time.size else None,
            "frame_p95": (
                float(np.percentile(frame_time, 95)) if frame_time.size else None
            ),
            "deadline_misses": (
                int((latency + frame_time > 1000.0 * unwrapped.step_dt).sum())
                if latency.size
                else 0
            ),
        },
    }

    output_path = Path(args_cli.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    env.close()


if __name__ == "__main__":
    main()
