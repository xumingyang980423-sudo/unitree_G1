"""Franka keyboard teleoperation - standalone, runs with system Python."""
import argparse, os, sys

_ISAAC_SIM_PATH = os.path.join("E:", os.sep, "Issac_sim", "isaac-sim-standalone-5.1.0-windows-x86_64")

os.add_dll_directory(os.path.join(_ISAAC_SIM_PATH, "kit", "python", "Lib", "site-packages", "h5py"))
import h5py; del h5py

os.environ.setdefault("CARB_APP_PATH", os.path.join(_ISAAC_SIM_PATH, "kit"))
os.environ.setdefault("ISAAC_PATH", _ISAAC_SIM_PATH)
os.environ.setdefault("EXP_PATH", os.path.join(_ISAAC_SIM_PATH, "apps"))

_exts_base = os.path.join(_ISAAC_SIM_PATH, "exts")
_sim_app_path = os.path.join(_exts_base, "isaacsim.simulation_app")
if _sim_app_path not in sys.path:
    sys.path.insert(0, _sim_app_path)

_isaaclab_source = os.path.join("E:", os.sep, "Issac_sim", "IsaacLab", "source")
for _d in os.listdir(_isaaclab_source):
    _p = os.path.join(_isaaclab_source, _d)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import isaacsim
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-IK-Rel-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.headless = False
args_cli.num_envs = 1

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym, torch, isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab_tasks.manager_based.manipulation.lift import mdp as lift_mdp
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
env_cfg.terminations.time_out = None
if hasattr(env_cfg.commands, "object_pose"):
    env_cfg.commands.object_pose.resampling_time_range = (1e9, 1e9)
    env_cfg.terminations.object_reached_goal = DoneTerm(func=lift_mdp.object_reached_goal)

print(f"[INFO] Task: {args_cli.task} | Device: {args_cli.device}")
env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

teleop = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.05, rot_sensitivity=0.05))
teleop.add_callback("R", lambda: (env.reset(), teleop.reset()))

print("=" * 60)
print("  Teleoperation Controls (Franka Robot):")
print("    W/S : Move X-axis (forward/back)")
print("    A/D : Move Y-axis (left/right)")
print("    Q/E : Move Z-axis (up/down)")
print("    Z/X : Rotate X  |  T/G : Rotate Y  |  C/V : Rotate Z")
print("    K   : Toggle gripper")
print("    R   : Reset environment")
print("=" * 60)

env.reset()
teleop.reset()

while simulation_app.is_running():
    action = teleop.advance()
    env.step(action.unsqueeze(0))

env.close()
simulation_app.close()
