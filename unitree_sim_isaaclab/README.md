# G1 Inspire 抓取抬起（PPO 强化学习）

Windows + Isaac Lab 上训练 G1 Inspire 五指手抓取 6cm 红块并抬起 5cm。

## 快速开始

```bat
run_train.bat          :: PPO 训练
run_play.bat           :: 加载 checkpoint 演示
view_scene.bat         :: 查看场景
run_train_demo.bat     :: 单 env 试跑
```

Checkpoint 目录：`logs/g1_grasp_lift_v4/checkpoints/`（四阶段奖励：靠近 → 手指动 → 抓牢 → 抬起）

## 文档

- **[G1_GRASP_NOTES.md](./G1_GRASP_NOTES.md)** — 方案踩坑、Franka vs G1、Pinocchio/WSL 限制、WSL 安装与官方 PINK IK 遥操作步骤
- **[TELEOP_SESSION.md](./TELEOP_SESSION.md)** — 键盘遥操作 Session 总结（conda、按键、文件、Pink IK 锁定踩坑）

## 键盘遥操作（Windows + conda）

```powershell
conda activate env_isaaclab
cd unitree_sim_isaaclab
.\run_teleop.bat
```

详见 **TELEOP_SESSION.md**。

## 说明

- **主路线**：PPO 关节控制（`env_cfg.py`），不依赖 Pinocchio。
- **键盘遥操作 / 模仿学习**：需 WSL/Linux + Pinocchio，见 `G1_GRASP_NOTES.md` 第 5 节。
- Franka 参考示例见仓库根目录 `franka_teleop/`（G1 部分同样需 Linux）。
