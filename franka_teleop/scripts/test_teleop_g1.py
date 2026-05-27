"""G1 keyboard teleop - absolute wrist poses + Inspire hand (Pink IK / Pinocchio)."""
import argparse
import os
import sys

_ISAAC_SIM_PATH = os.path.join("E:", os.sep, "Issac_sim", "isaac-sim-standalone-5.1.0-windows-x86_64")
os.add_dll_directory(os.path.join(_ISAAC_SIM_PATH, "kit", "python", "Lib", "site-packages", "h5py"))
import h5py  # noqa: F401

del h5py

os.environ.setdefault("CARB_APP_PATH", os.path.join(_ISAAC_SIM_PATH, "kit"))
os.environ.setdefault("ISAAC_PATH", _ISAAC_SIM_PATH)
os.environ.setdefault("EXP_PATH", os.path.join(_ISAAC_SIM_PATH, "apps"))
_exts_base = os.path.join(_ISAAC_SIM_PATH, "exts")
_sim_app_path = os.path.join(_exts_base, "isaacsim.simulation_app")
if _sim_app_path not in sys.path:
    sys.path.insert(0, _sim_app_path)

_pinocchio_deps = os.path.join(
    _ISAAC_SIM_PATH, "extscache", "omni.usd.libs-1.0.1+69cbf6ad.wx64.r.cp311", "bin"
)
if os.path.isdir(_pinocchio_deps):
    os.add_dll_directory(_pinocchio_deps)

parser = argparse.ArgumentParser(description="G1 Inspire hand keyboard teleop.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-PickPlace-G1-InspireFTP-Abs-v0",
    help="Isaac Lab task name.",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=True,
    help="Import Pinocchio before launching the simulator (required for Pink IK).",
)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

args_cli.headless = False
args_cli.num_envs = 1

import isaacsim  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab_tasks.utils import parse_env_cfg

TASK = args_cli.task
print(f"[INFO] Creating task: {TASK}")

env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=1)
env_cfg.terminations.time_out = None

env = gym.make(TASK, cfg=env_cfg).unwrapped
teleop = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.04, rot_sensitivity=0.08))

HAND_OPEN = [-0.5] * 12
HAND_CLOSE = [0.8] * 12

left_pos = np.array([-0.1487, 0.2038, 1.0952], dtype=np.float64)
left_quat = np.array([0.707, 0.0, 0.0, 0.707], dtype=np.float64)
right_pos = np.array([0.1487, 0.2038, 1.0952], dtype=np.float64)
right_quat = np.array([0.707, 0.0, 0.0, 0.707], dtype=np.float64)
gripper_closed = False


def reset_teleop_state() -> None:
    """Reset env and restore default wrist / hand command state."""
    global gripper_closed
    env.reset()
    teleop.reset()
    left_pos[:] = [-0.1487, 0.2038, 1.0952]
    left_quat[:] = [0.707, 0.0, 0.0, 0.707]
    right_pos[:] = [0.1487, 0.2038, 1.0952]
    right_quat[:] = [0.707, 0.0, 0.0, 0.707]
    gripper_closed = False
    print("[INFO] Environment reset.")


teleop.add_callback("R", reset_teleop_state)

print("=" * 60)
print("  G1 Inspire Hand - Keyboard Teleop")
print("    W/S: X  A/D: Y  Q/E: Z  Z/X/T/G/C/V: rotate")
print("    K: toggle grip  R: reset")
print("=" * 60)

reset_teleop_state()

while simulation_app.is_running():
    cmd = teleop.advance().cpu().numpy()
    dx, dy, dz, droll, dpitch, dyaw, gripper = cmd.astype(np.float64)

    right_pos += [dx, dy, dz]
    delta_r = R.from_euler("xyz", [droll, dpitch, dyaw])
    current_r = R.from_quat([right_quat[1], right_quat[2], right_quat[3], right_quat[0]])
    new_r = current_r * delta_r
    q = new_r.as_quat()
    right_quat[:] = [q[3], q[0], q[1], q[2]]

    if gripper > 0.5 and not gripper_closed:
        gripper_closed = True
    elif gripper < 0.5 and gripper_closed:
        gripper_closed = False

    rhand = HAND_CLOSE if gripper_closed else HAND_OPEN
    action = np.concatenate([left_pos, left_quat, right_pos, right_quat, HAND_OPEN, rhand]).astype(np.float32)
    env.step(torch.from_numpy(action).unsqueeze(0).to(env.device))

env.close()
simulation_app.close()
