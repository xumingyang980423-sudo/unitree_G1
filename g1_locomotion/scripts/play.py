"""G1 locomotion play - loads trained model directly."""
import argparse, os, sys, torch

_ISAAC_SIM_PATH = os.path.join("E:", os.sep, "Issac_sim", "isaac-sim-standalone-5.1.0-windows-x86_64")
_h5py_path = os.path.join(_ISAAC_SIM_PATH, "kit", "python", "Lib", "site-packages", "h5py")
os.add_dll_directory(_h5py_path)
import h5py; del h5py

os.environ.setdefault("CARB_APP_PATH", os.path.join(_ISAAC_SIM_PATH, "kit"))
os.environ.setdefault("ISAAC_PATH", _ISAAC_SIM_PATH)
os.environ.setdefault("EXP_PATH", os.path.join(_ISAAC_SIM_PATH, "apps"))
_exts_base = os.path.join(_ISAAC_SIM_PATH, "exts")
_sim_app_path = os.path.join(_exts_base, "isaacsim.simulation_app")
if _sim_app_path not in sys.path: sys.path.insert(0, _sim_app_path)

import isaacsim
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-G1-Play-v0")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym, isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
print(f"[INFO] Task: {args_cli.task} | Envs: {env_cfg.scene.num_envs}")

env = gym.make(args_cli.task, cfg=env_cfg)

checkpoint = args_cli.checkpoint
if checkpoint is None:
    import glob
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
    candidates = sorted(glob.glob(os.path.join(log_dir, "*", "checkpoints", "agent_*.pt")))
    if not candidates:
        candidates = sorted(glob.glob(os.path.join(log_dir, "checkpoints", "agent_*.pt")))
    checkpoint = candidates[-1] if candidates else None
    if not checkpoint:
        raise FileNotFoundError(f"No checkpoint in {log_dir}")

ckpt = torch.load(checkpoint, map_location=args_cli.device)
policy_sd = ckpt["policy"]

model = torch.nn.Sequential(
    torch.nn.Linear(123, 256), torch.nn.ELU(),
    torch.nn.Linear(256, 128), torch.nn.ELU(),
    torch.nn.Linear(128, 128), torch.nn.ELU(),
    torch.nn.Linear(128, 37),
)
model.load_state_dict({k.replace("net_container.", ""): v for k, v in policy_sd.items() if k.startswith("net_container.")})
model.to(args_cli.device)
model.eval()

print(f"[INFO] Loaded: {checkpoint}")

obs_dict, _ = env.reset()
obs_tensor = obs_dict["policy"]
with torch.inference_mode():
    while simulation_app.is_running():
        action = model(obs_tensor)
        obs_dict, _, dones, _, _ = env.step(action)
        obs_tensor = obs_dict["policy"]
        if dones:
            obs_dict, _ = env.reset()
            obs_tensor = obs_dict["policy"]

env.close()
simulation_app.close()
