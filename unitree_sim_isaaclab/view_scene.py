"""View G1 Inspire Hand + Red Block scene with zero-action hold."""
import os
import sys

_project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PROJECT_ROOT"] = _project_root

_ISAAC_SIM_PATH = "E:\\Issac_sim\\isaac-sim-standalone-5.1.0-windows-x86_64"
os.add_dll_directory(os.path.join(_ISAAC_SIM_PATH, "kit", "python", "Lib", "site-packages", "h5py"))
import h5py

del h5py
os.environ.setdefault("CARB_APP_PATH", os.path.join(_ISAAC_SIM_PATH, "kit"))
os.environ.setdefault("ISAAC_PATH", _ISAAC_SIM_PATH)
os.environ.setdefault("EXP_PATH", os.path.join(_ISAAC_SIM_PATH, "apps"))

import isaacsim
import torch
from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, _project_root)
from env_cfg import make_env_cfg
from isaaclab.envs import ManagerBasedRLEnv

print("[INFO] Building G1 Inspire Hand + Red Block scene...")
env_cfg = make_env_cfg(args_cli.num_envs, sim_device=args_cli.device)
env = ManagerBasedRLEnv(cfg=env_cfg)
env.reset()

act_dim = env.action_space.shape[-1]
zero_action = torch.zeros(env.num_envs, act_dim, device=env.device)

print("=" * 60)
print("  G1 + Inspire 5-Finger Hand + Table + Red Block")
print("  Holding default pose (zero delta actions)")
print("  Right-drag to orbit, scroll to zoom")
print("  Close window to exit")
print("=" * 60)

while simulation_app.is_running():
    # Step with zero actions so the robot holds its default pose instead of drifting.
    env.step(zero_action)

env.close()
simulation_app.close()
