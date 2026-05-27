"""Train G1 Inspire Hand pick-place - matches Unitree official config."""
import os, sys

_project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PROJECT_ROOT"] = _project_root

_ISAAC_SIM_PATH = "E:\\Issac_sim\\isaac-sim-standalone-5.1.0-windows-x86_64"
os.add_dll_directory(os.path.join(_ISAAC_SIM_PATH, "kit", "python", "Lib", "site-packages", "h5py"))
import h5py; del h5py
os.environ.setdefault("CARB_APP_PATH", os.path.join(_ISAAC_SIM_PATH, "kit"))
os.environ.setdefault("ISAAC_PATH", _ISAAC_SIM_PATH)
os.environ.setdefault("EXP_PATH", os.path.join(_ISAAC_SIM_PATH, "apps"))

import isaacsim
from isaaclab.app import AppLauncher
import argparse, torch
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--train_iters", type=int, default=2000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--checkpoint_interval", type=int, default=1000)
parser.add_argument("--resume", type=str, default=None, help="load weights from a previous agent_*.pt checkpoint")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

print(f"[INFO] G1 Inspire grasp-lift | Envs: {args_cli.num_envs}")

sys.path.insert(0, _project_root)
from env_cfg import make_env_cfg
from isaaclab.envs import ManagerBasedRLEnv

import skrl
from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.envs.wrappers.torch import wrap_env
from skrl.trainers.torch.sequential import SequentialTrainer
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.utils.model_instantiators.torch import gaussian_model, deterministic_model
from skrl.memories.torch import RandomMemory

env_cfg = make_env_cfg(args_cli.num_envs, sim_device=args_cli.device)
env = ManagerBasedRLEnv(cfg=env_cfg)
env = wrap_env(env)

obs_dim = env.observation_space.shape[-1]
act_dim = env.action_space.shape[-1]
print(f"[INFO] obs_dim={obs_dim} act_dim={act_dim}")

agent_cfg = PPO_CFG()
agent_cfg.rollouts = 16
agent_cfg.learning_epochs = 5
agent_cfg.mini_batches = 4
agent_cfg.discount_factor = 0.99
agent_cfg.gae_lambda = 0.95
agent_cfg.learning_rate = 3e-4
agent_cfg.learning_rate_scheduler = KLAdaptiveLR
agent_cfg.learning_rate_scheduler_kwargs = {"kl_threshold": 0.01}
agent_cfg.grad_norm_clip = 1.0
agent_cfg.ratio_clip = 0.2
agent_cfg.value_clip = 0.2
agent_cfg.entropy_loss_scale = 0.005
agent_cfg.value_loss_scale = 1.0

memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=args_cli.device)
log_dir = os.path.join(_project_root, "logs")
os.makedirs(log_dir, exist_ok=True)
experiment_name = "g1_grasp_lift_v7"
agent_cfg.experiment.directory = log_dir
agent_cfg.experiment.experiment_name = experiment_name
agent_cfg.experiment.checkpoint_interval = args_cli.checkpoint_interval

ckpt_dir = os.path.join(log_dir, experiment_name, "checkpoints")
total_timesteps = args_cli.train_iters * env.num_envs * agent_cfg.rollouts
print("=" * 60)
print(f"  Parallel envs (num_envs) : {args_cli.num_envs}")
print(f"  Training iterations      : {args_cli.train_iters}")
print(f"  Rollouts per iteration   : {agent_cfg.rollouts}")
print(f"  Total env steps          : {total_timesteps}")
print(f"  Checkpoint interval      : every {agent_cfg.experiment.checkpoint_interval} timesteps")
print(f"  Checkpoint save dir      : {ckpt_dir}")
print("=" * 60)
print(f"[INFO] Starting training...")

models = {
    "policy": gaussian_model(
        observation_space=env.observation_space, action_space=env.action_space,
        device=args_cli.device,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-3.0,
        max_log_std=-2.0,
        network=[{"name": "net", "input": "OBSERVATIONS", "layers": [512, 256, 128], "activations": "elu"}],
        output="ACTIONS",
    ),
    "value": deterministic_model(
        observation_space=env.observation_space, action_space=env.action_space,
        device=args_cli.device, clip_actions=False,
        network=[{"name": "net", "input": "OBSERVATIONS", "layers": [512, 256, 128], "activations": "elu"}], output="ONE",
    ),
}

agent = PPO(models=models, memory=memory, cfg=agent_cfg,
            observation_space=env.observation_space, action_space=env.action_space, device=args_cli.device)
agent.init()
if args_cli.resume:
    agent.load(args_cli.resume)
    print(f"[INFO] Resumed policy weights from: {args_cli.resume}")
trainer = SequentialTrainer(cfg={"timesteps": total_timesteps, "environment_info": "log"}, env=env, agents=[agent])
trainer.train()
env.close()
simulation_app.close()
