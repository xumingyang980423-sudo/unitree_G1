# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
G1 崎岖地形行走演示脚本 - Isaac-Velocity-Rough-G1-Play-v0

加载训练好的策略并运行演示。

启动方式:
    python scripts/play.py --task Isaac-Velocity-Rough-G1-Play-v0
    python scripts/play.py --task Isaac-Velocity-Rough-G1-Play-v0 --checkpoint logs/rsl_rl/.../model_3000.pt
"""

import argparse
import os
import sys

# --- Fix h5py DLL loading ---
_h5py_path = os.path.join("E:", os.sep, "isaac_sim", "isaac-sim-standalone-5.1.0-windows-x86_64",
                          "kit", "python", "Lib", "site-packages", "h5py")
os.add_dll_directory(_h5py_path)
import h5py; del h5py

# --- Patch isaacsim namespace ---
_exts_base = os.path.join("E:", os.sep, "isaac_sim", "isaac-sim-standalone-5.1.0-windows-x86_64", "exts")
_sim_app_path = os.path.join(_exts_base, "isaacsim.simulation_app")
if _sim_app_path not in sys.path:
    sys.path.insert(0, _sim_app_path)

import isaacsim
import isaacsim.simulation_app
isaacsim.SimulationApp = isaacsim.simulation_app.SimulationApp
isaacsim.__file__ = isaacsim.simulation_app.__file__

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play G1 humanoid locomotion with a trained policy.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-G1-Play-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunner, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg


def find_checkpoint(log_root: str) -> str | None:
    if not os.path.isdir(log_root):
        return None
    try:
        return get_checkpoint_path(log_root, run_dir=".*", checkpoint="model_.*.pt")
    except (ValueError, IndexError):
        return None


def main():
    task_name = args_cli.task

    env_cfg = parse_env_cfg(task_name, device=args_cli.device, num_envs=args_cli.num_envs)
    print(f"[INFO] Task: {task_name}")
    print(f"[INFO] Device: {args_cli.device}")
    print(f"[INFO] Number of envs: {env_cfg.scene.num_envs}")

    agent_cfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")

    checkpoint = args_cli.checkpoint
    if checkpoint is None:
        log_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "rsl_rl", task_name))
        checkpoint = find_checkpoint(log_root)
        if checkpoint is None:
            train_task = task_name.replace("-Play-v0", "-v0")
            train_log_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "rsl_rl", train_task))
            checkpoint = find_checkpoint(train_log_root)

    if checkpoint is None:
        raise FileNotFoundError("No checkpoint found. Train first or specify --checkpoint.")

    print(f"[INFO] Checkpoint: {checkpoint}")

    env = gym.make(task_name, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    env.seed(args_cli.seed)

    log_dir = os.path.dirname(checkpoint)
    runner = RslRlOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=args_cli.device)
    runner.load(checkpoint, checkpoint)
    print("[INFO] Policy loaded successfully.")

    obs = env.get_observations()
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = runner.policy(obs)
            obs, _, dones, _ = env.step(actions)


if __name__ == "__main__":
    main()
    simulation_app.close()
