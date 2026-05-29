# G1 Inspire 抓取抬起（PPO 强化学习 + 键盘遥操作）

Windows + Isaac Lab 上训练 G1 Inspire 五指手抓取红块/圆柱并抬起；支持键盘 Pink IK 遥操作。

## 快速开始

在仓库根目录执行（`run_*.bat` 会转发到 `scripts/`）：

```bat
run_train.bat          :: PPO 训练
run_play.bat           :: 加载 checkpoint 演示
view_scene.bat         :: 查看场景
run_train_demo.bat     :: 单 env 试跑
run_teleop.bat         :: 键盘遥操作（需 conda env_isaaclab + pinocchio）
```

Checkpoint 目录：`logs/g1_grasp_lift_v7/checkpoints/`（以你当前训练版本为准）

## 目录结构

详见 **[docs/PROJECT_LAYOUT.md](./docs/PROJECT_LAYOUT.md)**。

| 目录 | 内容 |
|------|------|
| `grasp_rl/` | PPO 环境、训练、评估 |
| `teleop/` | 键盘遥操作 |
| `scripts/` | 启动脚本 |
| `docs/` | 项目文档 |

## 文档

- **[docs/G1_GRASP_NOTES.md](./docs/G1_GRASP_NOTES.md)** — 方案踩坑、Franka vs G1、Pinocchio/WSL
- **[docs/TELEOP_SESSION.md](./docs/TELEOP_SESSION.md)** — 键盘遥操作（conda、按键、Pink IK）
- **[docs/IL_TELEOP_TO_IMITATION_PLAN.md](./docs/IL_TELEOP_TO_IMITATION_PLAN.md)** — 录演示 → 模仿学习

## 键盘遥操作

```powershell
conda activate env_isaaclab
cd unitree_sim_isaaclab
.\run_teleop.bat
```

详见 **docs/TELEOP_SESSION.md**。

## 说明

- **主路线**：PPO 关节控制（`grasp_rl/env_cfg.py`），不依赖 Pinocchio。
- **键盘遥操作**：`teleop/teleop_grasp.py`，需 conda `env_isaaclab` + Pinocchio。
- 上游 Unitree 整机仿真：`sim_main.py`、`tasks/`、`doc/`（安装说明）。
