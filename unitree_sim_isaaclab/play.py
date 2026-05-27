"""Play trained G1 Inspire pick-place policy with action smoothing."""
import argparse
import glob
import os
import sys

import torch

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
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--smooth", type=float, default=1.0, help="action EMA alpha (1.0=no smoothing)")
parser.add_argument("--deterministic", action="store_true", default=True, help="use policy mean (no sampling noise)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, _project_root)
from env_cfg import make_env_cfg
from isaaclab.envs import ManagerBasedRLEnv

env_cfg = make_env_cfg(args_cli.num_envs, sim_device=args_cli.device)
env = ManagerBasedRLEnv(cfg=env_cfg)

checkpoint = args_cli.checkpoint
if checkpoint is not None:
    # Strip invisible bidi / zero-width chars from copy-paste (causes OSError 22 on Windows).
    for ch in ("\u202a", "\u202b", "\u202c", "\u202d", "\u202e", "\u200e", "\u200f", "\ufeff"):
        checkpoint = checkpoint.replace(ch, "")
    checkpoint = checkpoint.strip().strip('"')
if checkpoint is None:
    log_dir = os.path.join(_project_root, "logs")
    for sub in ("g1_grasp_lift_v7", "g1_grasp_lift_v6", "g1_grasp_lift_v5", "g1_grasp_lift_v4", "g1_grasp_lift_v3", "g1_grasp_lift_v2", "g1_grasp_lift", "g1_pickplace_v2", "g1_pickplace"):
        candidates = sorted(glob.glob(os.path.join(log_dir, sub, "checkpoints", "agent_*.pt")))
        if candidates:
            checkpoint = candidates[-1]
            break
    if checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found under {log_dir}")

ckpt = torch.load(checkpoint, map_location=args_cli.device, weights_only=False)
policy_sd = ckpt["policy"]
print(f"[INFO] Loaded checkpoint: {checkpoint}")

obs_dict, _ = env.reset()
obs_tensor = obs_dict["policy"]
obs_dim = obs_tensor.shape[-1]
act_dim = env.action_space.shape[-1]
print(f"[INFO] obs_dim={obs_dim} act_dim={act_dim}")
print(f"[INFO] Play mode: deterministic={args_cli.deterministic}, smooth={args_cli.smooth}")

model = torch.nn.Sequential(
    torch.nn.Linear(obs_dim, 512),
    torch.nn.ELU(),
    torch.nn.Linear(512, 256),
    torch.nn.ELU(),
    torch.nn.Linear(256, 128),
    torch.nn.ELU(),
    torch.nn.Linear(128, act_dim),
)
model.load_state_dict(
    {k.replace("net_container.", ""): v for k, v in policy_sd.items() if k.startswith("net_container.")},
    strict=False,
)
model.to(args_cli.device)
model.eval()

prev_action = torch.zeros(args_cli.num_envs, act_dim, device=args_cli.device)
alpha = args_cli.smooth

with torch.inference_mode():
    while simulation_app.is_running():
        raw_action = model(obs_tensor)
        action = alpha * raw_action + (1.0 - alpha) * prev_action
        prev_action = action.clone()
        obs_dict, _, dones, _, _ = env.step(action)
        obs_tensor = obs_dict["policy"]
        if dones.any():
            obs_dict, _ = env.reset()
            obs_tensor = obs_dict["policy"]
            prev_action.zero_()

env.close()
simulation_app.close()
