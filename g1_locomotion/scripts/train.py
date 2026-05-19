# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
G1 崎岖地形行走训练脚本 - Isaac-Velocity-Rough-G1-v0

启动方式:
    python scripts/train.py --task Isaac-Velocity-Rough-G1-v0
    python scripts/train.py --task Isaac-Velocity-Rough-G1-v0 --headless
"""

import argparse
import os
import sys

# --- Fix h5py DLL loading: h5py's hdf5.dll conflicts with Kit's sensor HDF5 DLL ---
# Must import h5py BEFORE Kit loads, otherwise Kit's HDF5 grabs the DLL name first.
_h5py_path = os.path.join("E:", os.sep, "isaac_sim", "isaac-sim-standalone-5.1.0-windows-x86_64",
                          "kit", "python", "Lib", "site-packages", "h5py")
os.add_dll_directory(_h5py_path)
import h5py; del h5py  # pre-load h5py's DLLs into memory

# --- Patch isaacsim namespace for AppLauncher ---
# Isaac Lab expects `from isaacsim import SimulationApp`, but isaacsim is a namespace
# package (no __init__.py). Patch SimulationApp onto the namespace and fix __file__.
_exts_base = os.path.join("E:", os.sep, "isaac_sim", "isaac-sim-standalone-5.1.0-windows-x86_64", "exts")
_sim_app_path = os.path.join(_exts_base, "isaacsim.simulation_app")
if _sim_app_path not in sys.path:
    sys.path.insert(0, _sim_app_path)

import isaacsim  # namespace package
import isaacsim.simulation_app  # loads SimulationApp into submodule
isaacsim.SimulationApp = isaacsim.simulation_app.SimulationApp  # expose at top level
isaacsim.__file__ = isaacsim.simulation_app.__file__  # fix namespace __file__

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train G1 humanoid locomotion on rough terrain.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-G1-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments (default: from config).")
parser.add_argument("--max_iterations", type=int, default=None, help="Max training iterations.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunner, RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg


def main():
    task_name = args_cli.task

    env_cfg = parse_env_cfg(task_name, device=args_cli.device, num_envs=args_cli.num_envs)
    print(f"[INFO] Task: {task_name}")
    print(f"[INFO] Device: {args_cli.device}")
    print(f"[INFO] Number of envs: {env_cfg.scene.num_envs}")

    agent_cfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    print(f"[INFO] Max iterations: {agent_cfg.max_iterations}")
    print(f"[INFO] Experiment name: {agent_cfg.experiment_name}")

    log_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "rsl_rl", task_name))
    os.makedirs(log_root, exist_ok=True)
    print(f"[INFO] Log directory: {log_root}")

    env = gym.make(task_name, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    env.seed(args_cli.seed)

    runner = RslRlOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_root, device=args_cli.device)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
