"""Dexsuite Kuka+Allegro dexterous reorient RL training (SKRL PPO)."""
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
parser.add_argument("--task", type=str, default="Isaac-Dexsuite-Kuka-Allegro-Reorient-v0")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--train_iters", type=int, default=5000)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym, torch, isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg

import skrl
from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.envs.wrappers.torch import wrap_env
from skrl.trainers.torch.sequential import SequentialTrainer
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.utils.model_instantiators.torch import gaussian_model, deterministic_model
from skrl.memories.torch import RandomMemory

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
print(f"[INFO] Task: {args_cli.task} | Device: {args_cli.device} | Envs: {env_cfg.scene.num_envs}")

env = gym.make(args_cli.task, cfg=env_cfg)
env = wrap_env(env)

agent_cfg = PPO_CFG()
agent_cfg.rollouts = 16
agent_cfg.learning_epochs = 5
agent_cfg.mini_batches = 4
agent_cfg.discount_factor = 0.99
agent_cfg.gae_lambda = 0.95
agent_cfg.learning_rate = 5e-4
agent_cfg.learning_rate_scheduler = KLAdaptiveLR
agent_cfg.learning_rate_scheduler_kwargs = {"kl_threshold": 0.01}
agent_cfg.grad_norm_clip = 1.0
agent_cfg.ratio_clip = 0.2
agent_cfg.value_clip = 0.2
agent_cfg.entropy_loss_scale = 0.01
agent_cfg.value_loss_scale = 1.0

memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=args_cli.device)

log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "dexsuite_reorient"))
os.makedirs(log_dir, exist_ok=True)
agent_cfg.experiment.directory = log_dir
agent_cfg.experiment.experiment_name = "dexsuite_reorient"
agent_cfg.experiment.checkpoint_interval = 500

print(f"[INFO] Training {args_cli.train_iters} iterations")

models = {
    "policy": gaussian_model(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=args_cli.device,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-20.0,
        max_log_std=2.0,
        network=[{"name": "net", "input": "OBSERVATIONS", "layers": [1024, 512, 256], "activations": "elu"}],
        output="ACTIONS",
    ),
    "value": deterministic_model(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=args_cli.device,
        clip_actions=False,
        network=[{"name": "net", "input": "OBSERVATIONS", "layers": [1024, 512, 256], "activations": "elu"}],
        output="ONE",
    ),
}

agent = PPO(
    models=models, memory=memory, cfg=agent_cfg,
    observation_space=env.observation_space, action_space=env.action_space, device=args_cli.device
)
agent.init()
print("[INFO] Agent initialized")

total_timesteps = args_cli.train_iters * env.num_envs * agent_cfg.rollouts
trainer = SequentialTrainer(cfg={"timesteps": total_timesteps, "environment_info": "log"}, env=env, agents=[agent])
trainer.train()

env.close()
simulation_app.close()
