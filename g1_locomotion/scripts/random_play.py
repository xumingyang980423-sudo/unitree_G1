# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
G1 崎岖地形行走随机演示脚本 - Isaac-Velocity-Rough-G1-Play-v0

用随机动作控制 G1（无需训练好的策略），用于快速验证环境和观察场景。

启动方式:
    python scripts/random_play.py --task Isaac-Velocity-Rough-G1-Play-v0
    python scripts/random_play.py --task Isaac-Velocity-Rough-G1-Play-v0 --num_envs 4
"""

import argparse
import os
import sys

# --- Fix h5py DLL loading (must load h5py BEFORE Kit loads conflicting HDF5 DLLs) ---
_h5py_path = os.path.join("E:", os.sep, "isaac_sim", "isaac-sim-standalone-5.1.0-windows-x86_64",
                          "kit", "python", "Lib", "site-packages", "h5py")
os.add_dll_directory(_h5py_path)
import h5py  # force-load h5py's hdf5.dll before Kit loads its own
del h5py

# --- Patch isaacsim namespace for AppLauncher ---
_exts_base = os.path.join("E:", os.sep, "isaac_sim", "isaac-sim-standalone-5.1.0-windows-x86_64", "exts")
_sim_app_path = os.path.join(_exts_base, "isaacsim.simulation_app")
if _sim_app_path not in sys.path:
    sys.path.insert(0, _sim_app_path)

import isaacsim
import isaacsim.simulation_app
isaacsim.SimulationApp = isaacsim.simulation_app.SimulationApp
isaacsim.__file__ = isaacsim.simulation_app.__file__

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Quick test of G1 locomotion env with random actions.")
parser.add_argument(
    "--task", type=str, default="Isaac-Velocity-Rough-G1-Play-v0", help="Task name."
)
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments.")
parser.add_argument("--episode_length", type=int, default=500, help="Steps per episode.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    task_name = args_cli.task

    # --- 1. 加载环境配置 ---
    env_cfg = parse_env_cfg(task_name, device=args_cli.device, num_envs=args_cli.num_envs)
    print(f"[INFO] Task: {task_name}")
    print(f"[INFO] Device: {args_cli.device}")
    print(f"[INFO] Number of envs: {env_cfg.scene.num_envs}")

    # --- 2. 创建环境 ---
    env = gym.make(task_name, cfg=env_cfg)
    obs, _ = env.reset()

    # --- 3. 用随机动作运行 ---
    step_count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            action = torch.rand(env.action_space.shape, device=env.unwrapped.device) * 2.0 - 1.0
            obs, _, terminated, truncated, _ = env.step(action)

            step_count += 1
            if terminated.any() or truncated.any() or step_count >= args_cli.episode_length:
                obs, _ = env.reset()
                step_count = 0
                print("[INFO] Episode reset.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
