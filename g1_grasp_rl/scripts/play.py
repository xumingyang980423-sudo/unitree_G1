"""G1 grasp-and-lift play - loads trained model."""
import argparse, glob, os, sys, torch

_ISAAC_SIM_PATH = "E:\\Issac_sim\\isaac-sim-standalone-5.1.0-windows-x86_64"
os.add_dll_directory(os.path.join(_ISAAC_SIM_PATH, "kit", "python", "Lib", "site-packages", "h5py"))
import h5py; del h5py

os.environ.setdefault("CARB_APP_PATH", os.path.join(_ISAAC_SIM_PATH, "kit"))
os.environ.setdefault("ISAAC_PATH", _ISAAC_SIM_PATH)
os.environ.setdefault("EXP_PATH", os.path.join(_ISAAC_SIM_PATH, "apps"))

import isaacsim
from isaaclab.app import AppLauncher

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from g1_grasp_env_cfg import G1GraspRLEnvCfg
from isaaclab.envs import ManagerBasedRLEnv

env_cfg = G1GraspRLEnvCfg()
env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.sim.device = "cpu"
print(f"[INFO] G1 Grasp Play | Envs: {args_cli.num_envs}")

env = ManagerBasedRLEnv(cfg=env_cfg)

checkpoint = args_cli.checkpoint
if checkpoint is None:
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
    candidates = sorted(glob.glob(os.path.join(log_dir, "**", "checkpoints", "agent_*.pt"), recursive=True))
    checkpoint = candidates[-1] if candidates else None
    if not checkpoint:
        raise FileNotFoundError(f"No checkpoint found in {log_dir}")

ckpt = torch.load(checkpoint, map_location=args_cli.device)
policy_sd = ckpt["policy"]
print(f"[INFO] Loaded: {checkpoint}")

obs_dict, _ = env.reset()
obs_tensor = obs_dict["policy"]
obs_dim = obs_tensor.shape[-1]
act_dim = env.action_space.shape[-1]
print(f"[INFO] obs_dim={obs_dim} act_dim={act_dim}")

model = torch.nn.Sequential(
    torch.nn.Linear(obs_dim, 512), torch.nn.ELU(),
    torch.nn.Linear(512, 256), torch.nn.ELU(),
    torch.nn.Linear(256, 128), torch.nn.ELU(),
    torch.nn.Linear(128, act_dim),
)
model.load_state_dict(
    {k.replace("net_container.", ""): v for k, v in policy_sd.items() if k.startswith("net_container.")},
    strict=False,
)
model.to(args_cli.device)
model.eval()

prev_action = torch.zeros(1, act_dim, device="cpu")
alpha = 0.3  # smoothing factor (0=no change, 1=instant)

with torch.inference_mode():
    while simulation_app.is_running():
        raw_action = model(obs_tensor)
        action = alpha * raw_action + (1 - alpha) * prev_action
        prev_action = action.clone()
        obs_dict, _, dones, _, _ = env.step(action)
        obs_tensor = obs_dict["policy"]
        if dones.any():
            obs_dict, _ = env.reset()
            obs_tensor = obs_dict["policy"]

env.close()
simulation_app.close()
