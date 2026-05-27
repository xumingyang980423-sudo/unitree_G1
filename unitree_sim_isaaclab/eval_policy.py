"""Headless policy eval: success rate and lift height (low VRAM)."""
import argparse
import glob
import os
import sys

_project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PROJECT_ROOT"] = _project_root

_ISAAC_SIM_PATH = os.environ.get(
    "ISAAC_SIM_PATH", r"E:\Issac_sim\isaac-sim-standalone-5.1.0-windows-x86_64"
)
os.add_dll_directory(os.path.join(_ISAAC_SIM_PATH, "kit", "python", "Lib", "site-packages", "h5py"))
import h5py  # noqa: F401

del h5py
os.environ.setdefault("CARB_APP_PATH", os.path.join(_ISAAC_SIM_PATH, "kit"))
os.environ.setdefault("ISAAC_PATH", _ISAAC_SIM_PATH)
os.environ.setdefault("EXP_PATH", os.path.join(_ISAAC_SIM_PATH, "apps"))

import isaacsim  # noqa: F401
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--max_steps", type=int, default=375, help="15s @ 25Hz")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

sys.path.insert(0, _project_root)
from env_cfg import LIFT_TARGET_Z, OBJECT_REST_Z, _right_finger_closure, _task_success, make_env_cfg
from isaaclab.envs import ManagerBasedRLEnv

env_cfg = make_env_cfg(args_cli.num_envs, sim_device=args_cli.device)
env_cfg.scene.replicate_physics = False
env = ManagerBasedRLEnv(cfg=env_cfg)

ckpt = torch.load(args_cli.checkpoint, map_location=args_cli.device, weights_only=False)
policy_sd = ckpt["policy"]
obs_dict, _ = env.reset()
obs_tensor = obs_dict["policy"]
obs_dim = obs_tensor.shape[-1]
act_dim = env.action_space.shape[-1]

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

successes = 0
max_z_list = []
max_closure_near_list = []
ep = 0
step = 0

print(f"[EVAL] checkpoint: {args_cli.checkpoint}")
print(f"[EVAL] episodes={args_cli.episodes} headless=True device={args_cli.device}")

with torch.inference_mode():
    while ep < args_cli.episodes and simulation_app.is_running():
        action = model(obs_tensor)
        obs_dict, _, dones, _, _ = env.step(action)
        obs_tensor = obs_dict["policy"]
        step += 1

        obj_z = env.scene["object"].data.root_pos_w[:, 0].item()
        if not hasattr(env, "_ep_max_z"):
            env._ep_max_z = obj_z
            env._ep_max_closure = _right_finger_closure(env)[0].item()
        env._ep_max_z = max(env._ep_max_z, obj_z)
        env._ep_max_closure = max(env._ep_max_closure, _right_finger_closure(env)[0].item())

        if dones.any() or step >= args_cli.max_steps:
            if _task_success(env)[0].item() or env._ep_max_z >= LIFT_TARGET_Z - 0.005:
                successes += 1
            max_z_list.append(env._ep_max_z)
            max_closure_near_list.append(env._ep_max_closure)
            ep += 1
            step = 0
            del env._ep_max_z
            del env._ep_max_closure
            obs_dict, _ = env.reset()
            obs_tensor = obs_dict["policy"]

env.close()
simulation_app.close()

n = len(max_z_list)
lift_thresh = LIFT_TARGET_Z
lift_count = sum(1 for z in max_z_list if z >= lift_thresh - 0.005)
print("=" * 60)
print(f"  Episodes evaluated     : {n}")
print(f"  Task success rate      : {successes}/{n} ({100.0 * successes / max(n, 1):.1f}%)")
print(f"  Block lifted >=5cm     : {lift_count}/{n} ({100.0 * lift_count / max(n, 1):.1f}%)")
print(f"  Mean max object Z      : {sum(max_z_list) / max(n, 1):.4f} m  (rest={OBJECT_REST_Z}, target={lift_thresh})")
print(f"  Mean max finger closure: {sum(max_closure_near_list) / max(n, 1):.3f}  (1.0=fully closed)")
print("=" * 60)
