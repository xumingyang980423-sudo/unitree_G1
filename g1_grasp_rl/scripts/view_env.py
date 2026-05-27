"""Quick view G1 grasp RL scene - creates env, resets, renders."""
import argparse, os, sys

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
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.num_envs = 1

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from g1_grasp_env_cfg import G1GraspRLEnvCfg
from isaaclab.envs import ManagerBasedRLEnv

env_cfg = G1GraspRLEnvCfg()
env_cfg.scene.num_envs = 1
env_cfg.sim.device = "cpu"
env = ManagerBasedRLEnv(cfg=env_cfg)

print("[INFO] Resetting scene...")
env.reset()

print("=" * 60)
print("  G1 + table + red cube (grasping scene)")
print("  Right-drag to orbit, scroll to zoom")
print("  Close window to exit")
print("=" * 60)

while simulation_app.is_running():
    env.sim.render()

env.close()
simulation_app.close()
