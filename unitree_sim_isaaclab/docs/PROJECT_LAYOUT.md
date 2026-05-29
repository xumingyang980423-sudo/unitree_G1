# 项目目录说明（unitree_sim_isaaclab）

仓库根目录即 `PROJECT_ROOT`（`assets/`、`robots/`、`tasks/`、`logs/` 仍在此层）。

## 顶层结构

```text
unitree_sim_isaaclab/
├── README.md                 # 本仓库 G1 抓取 / 遥操作入口
├── run_*.bat                 # 转发到 scripts/（兼容旧用法）
├── scripts/                  # Windows 启动脚本
├── docs/                     # 本项目文档（抓取、遥操作、IL 方案）
├── grasp_rl/                 # PPO 抓取抬起 RL
├── teleop/                   # 键盘 Pink IK 遥操作
├── assets/                   # USD 机器人与场景资源
├── robots/                   # 机器人 Articulation 配置
├── tasks/                    # Unitree 官方多任务环境
├── logs/                     # 训练 checkpoint（本地生成）
├── tools/                    # 数据与 USD 工具
├── doc/                      # Unitree 上游 Isaac Sim 安装文档
├── dds/                      # DDS 通信（整机仿真）
├── sim_main.py               # 整机 DDS 仿真入口
└── ...
```

## scripts/ — 启动脚本

| 脚本 | 调用的 Python 入口 |
|------|-------------------|
| `run_teleop.bat` | `teleop/teleop_grasp.py` |
| `run_train.bat` | `grasp_rl/train.py` |
| `run_play.bat` | `grasp_rl/play.py` |
| `run_eval.bat` | `grasp_rl/eval_policy.py` |
| `run_train_demo.bat` | `grasp_rl/train.py`（单 env 演示） |
| `view_scene.bat` | `grasp_rl/view_scene.py` |
| `install_cuda_torch.bat` | Isaac Lab 自带 Python 装 CUDA PyTorch |

环境变量：`REPO_ROOT` = `scripts/` 的上一级（仓库根）。

## grasp_rl/ — 强化学习抓取

| 文件 | 说明 |
|------|------|
| `env_cfg.py` | G1 Inspire 红块/抓取环境配置 |
| `coupled_grasp_action.py` | 四指 + 拇指联动动作 |
| `train.py` | PPO 训练 |
| `play.py` | 加载策略演示 |
| `eval_policy.py` | 无头评估成功率 |
| `view_scene.py` | 查看场景 |
| `_paths.py` | `PROJECT_ROOT` 与 `sys.path` |

## teleop/ — 键盘遥操作

| 文件 | 说明 |
|------|------|
| `teleop_grasp.py` | 主循环（Se3 键盘 + Pink IK） |
| `teleop_pink_env_cfg.py` | 圆柱场景、actuator、Pink 任务 |
| `teleop_fingers.py` | 12 关节直接 PD + 抓取锁定 |
| `teleop_hand.py` | Pink hand 向量 |
| `teleop_grasp_detect.py` | 抓取 / 抬起检测 |
| `teleop_contact_zone.py` | 对准区域提示 |

## docs/ — 文档

| 文件 | 说明 |
|------|------|
| `G1_GRASP_NOTES.md` | RL 踩坑、Pinocchio/WSL、官方 IK |
| `TELEOP_SESSION.md` | 键盘遥操作 Session 总结 |
| `IL_TELEOP_TO_IMITATION_PLAN.md` | 遥操作录数据 → 模仿学习 |
| `PROJECT_LAYOUT.md` | 本文件 |

## Python 导入约定

- 入口脚本将 **仓库根** 加入 `sys.path`，并设置 `PROJECT_ROOT`。
- 包内导入示例：
  - `from grasp_rl.env_cfg import make_env_cfg`
  - `from teleop.teleop_fingers import DirectFingerController`
- `assets`、`robots` 路径仍相对 `PROJECT_ROOT`，与重组前一致。

## 常用命令（不变）

```powershell
cd E:\unitreeG1\unitree_G1\unitree_sim_isaaclab
conda activate env_isaaclab
.\run_teleop.bat
.\run_train.bat
```
