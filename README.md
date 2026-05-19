# Unitree G1 - Isaac Sim 部署项目

基于 NVIDIA Isaac Sim 5.1.0 的 Unitree G1 人形机器人仿真训练与演示。

## 环境要求

- Windows 10/11
- NVIDIA GPU (RTX 系列推荐)
- [Isaac Sim 5.1.0](https://developer.nvidia.com/isaac-sim) 安装在 `E:\isaac_sim\`
- 网络连接（首次运行需从 NVIDIA Nucleus 下载 USD 资产）

## 项目结构

```
g1_locomotion/
├── run.bat                    # 一键启动（交互菜单）
├── logs/                      # 训练日志和模型检查点
└── scripts/
    ├── train.py               # 训练脚本
    ├── play.py                # 演示脚本（加载训练好的策略）
    └── random_play.py         # 快速测试（随机动作，无需训练）
```

## 快速开始

双击 `g1_locomotion\run.bat`，选择菜单操作：

| 选项 | 说明 |
|------|------|
| [3] Quick test | 随机动作测试，验证环境（无需 GPU 训练） |
| [1] Train | 开始 RL 训练 |
| [2] Play | 加载训练好的策略演示 |
| [4] Train (headless) | 无 GUI 训练 |

或命令行直接运行：

```bash
cd g1_locomotion
run.bat
```

## 场景说明

| 场景 ID | 说明 |
|----------|------|
| `Isaac-Velocity-Rough-G1-v0` | G1 崎岖地形行走（训练模式） |
| `Isaac-Velocity-Rough-G1-Play-v0` | G1 崎岖地形行走（演示模式） |

更多场景详见 `Unitree_G1_Scene_Cases.md`。

## 依赖

- Isaac Sim 自带的 Python 解释器 (`E:\isaac_sim\...\kit\python\python.exe`)
- isaacsim (Kit 扩展)
- isaaclab (已安装于 site-packages)
