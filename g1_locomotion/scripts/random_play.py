# random_play.py
import argparse, os, sys
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
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
import gymnasium as gym, torch, isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg
def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    print(f"[INFO] Task: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()
    step = 0
    while simulation_app.is_running():
        action = torch.rand(env.action_space.shape, device=env.unwrapped.device) * 2 - 1
        obs, _, term, trunc, _ = env.step(action)
        step += 1
        if term.any() or trunc.any() or step >= 500:
            obs, _ = env.reset(); step = 0
    env.close()
if __name__ == "__main__": main(); simulation_app.close()
