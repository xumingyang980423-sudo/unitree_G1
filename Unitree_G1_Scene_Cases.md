# Unitree G1 全部场景案例 (Isaac Sim 5.1.0)

> 基于 `E:\isaac_sim\isaac-sim-standalone-5.1.0-windows-x86_64` 安装路径分析整理

---

## 一、运动控制任务 (Locomotion - Velocity Tracking)

G1 双足行走，通过 RL 策略跟踪指令速度。

| 场景 ID | 说明 | 地形 |
|----------|------|------|
| `Isaac-Velocity-Rough-G1-v0` | G1 崎岖地形行走（训练模式） | 随机崎岖地形 |
| `Isaac-Velocity-Rough-G1-Play-v0` | G1 崎岖地形行走（演示/播放模式） | 随机崎岖地形（少量并行环境） |
| `Isaac-Velocity-Flat-G1-v0` | G1 平坦地形行走（训练模式） | 平面 |
| `Isaac-Velocity-Flat-G1-Play-v0` | G1 平坦地形行走（演示/播放模式） | 平面（少量并行环境） |

**运行方式:**
```bash
python scripts/rsl_rl/train.py --task Isaac-Velocity-Rough-G1-v0
python scripts/rsl_rl/play.py --task Isaac-Velocity-Rough-G1-v0
```

**关键配置:**
- 机器人: `G1_MINIMAL_CFG`（轻量碰撞模型）
- RL 算法: RSL-RL PPO / SKRL PPO
- 控制: 位置/速度控制（leg joints） + 力矩控制（arm joints）
- 观测: 关节角度、关节速度、基座线/角速度、指令速度、高度扫描（rough only）

---

## 二、操作任务 (Manipulation - Pick & Place)

G1 固定基座上身抓取放置任务。

| 场景 ID | 说明 |
|----------|------|
| `Isaac-PickPlace-G1-InspireFTP-Abs-v0` | G1 + Inspire 五指灵巧手 Pick & Place |

**运行方式:**
```bash
python scripts/robomimic/train.py --task Isaac-PickPlace-G1-InspireFTP-Abs-v0
```

**关键配置:**
- 机器人: `G1_INSPIRE_FTP_CFG`（29 DoF + Inspire 5指手）
- 场景: 桌面 + 物体（从上方抓取）
- 控制: PINK IK + NullSpacePosture
- 算法: Robomimic BC-RNN
- 目标: 绝对位置 (Abs)

---

## 三、移动操作任务 (Locomanipulation)

G1 同时行走和操作——全身协调的 Pick & Place。

| 场景 ID | 说明 |
|----------|------|
| `Isaac-PickPlace-Locomanipulation-G1-Abs-v0` | G1 全身行走+抓取放置 |
| `Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0` | G1 固定基座上身 IK 抓取放置 |

**运行方式:**
```bash
python scripts/robomimic/train.py --task Isaac-PickPlace-Locomanipulation-G1-Abs-v0
```

**关键配置:**
- 机器人: `G1_29DOF_CFG`
- 下半身: Agile RL Teacher Policy（运动学指令）
- 上半身: PINK IK 控制器
- 场景: 桌面 + 可抓取物体
- 支持 VR 遥操作采集数据

---

## 四、模仿学习任务 (Isaac Lab Mimic)

### 4.1 技能数据生成 (SDG)

| 场景 ID | 说明 |
|----------|------|
| `Isaac-G1-SteeringWheel-Locomanipulation` | G1 方向盘移动操作 SDG |

- 用于自动采集 G1 移动操作演示数据
- 生成训练数据集供模仿学习

### 4.2 Pinocchio 环境

| 场景 ID | 说明 |
|----------|------|
| `Isaac-Locomanipulation-G1-Abs-Mimic-v0` | G1 移动操作模仿学习（Pinocchio 运动学） |

- 基于 Pinocchio 的快速运动学仿真
- 用于策略部署和评估

---

## 五、VR 遥操作配置 (OpenXR Teleoperation)

支持通过 VR 设备（OpenXR）实时遥控 G1 机器人。

### 5.1 Trihand（三指灵巧手）遥操作

| 配置文件 | 功能 |
|----------|------|
| `g1_upper_body_retargeter.py` | G1 上半身重定向（Trihand） |
| `g1_upper_body_motion_ctrl_retargeter.py` | G1 上半身运动控制重定向 |
| `g1_upper_body_motion_ctrl_gripper.py` | G1 夹爪控制重定向 |
| `g1_dex_retargeting_utils.py` | DexPilot 灵巧手重定向工具 |

### 5.2 Inspire Hand（五指灵巧手）遥操作

| 配置文件 | 功能 |
|----------|------|
| `g1_upper_body_retargeter.py` | G1 上半身重定向（Inspire 5指手） |
| `g1_dex_retargeting_utils.py` | DexPilot 灵巧手重定向工具 |

### 5.3 下半身移动遥操作

| 配置文件 | 功能 |
|----------|------|
| `g1_lower_body_standing.py` | G1 站立姿态维持 |
| `g1_motion_controller_locomotion.py` | G1 VR 移动控制 |

---

## 六、G1 机器人模型配置

在 `isaaclab_assets/robots/unitree.py` 中定义：

| 配置名 | USD 路径 | DoF | 用途 |
|--------|----------|-----|------|
| `G1_CFG` | `{NUCLEUS}/Robots/Unitree/G1/g1.usd` | 23 | 标准 G1 人形机器人 |
| `G1_MINIMAL_CFG` | `{NUCLEUS}/Robots/Unitree/G1/g1_minimal.usd` | 23 | 轻量碰撞模型（提速） |
| `G1_29DOF_CFG` | `{NUCLEUS}/Robots/Unitree/G1/g1_29dof_inspire_hand.usd` | 29 | 移动操作配置 |
| `G1_INSPIRE_FTP_CFG` | `{NUCLEUS}/Robots/Unitree/G1/g1_29dof_inspire_hand.usd` | 29 | Inspire 五指手配置 |

> **注意:** USD 模型资产托管在 NVIDIA Nucleus 服务器，运行时需要网络连接。

---

## 七、相关文件索引

```text
# 机器人定义
isaaclab_assets/robots/unitree.py

# 运动任务
isaaclab_tasks/manager_based/locomotion/velocity/config/g1/
├── __init__.py               # 4 个 Gym 任务注册
├── rough_env_cfg.py          # 崎岖地形配置
├── flat_env_cfg.py           # 平地配置
└── agents/
    ├── rsl_rl_ppo_cfg.py
    ├── skrl_rough_ppo_cfg.yaml
    └── skrl_flat_ppo_cfg.yaml

# 操作任务
isaaclab_tasks/manager_based/manipulation/pick_place/
├── __init__.py               # G1 pick-place 注册
└── pickplace_unitree_g1_inspire_hand_env_cfg.py

# 移动操作任务
isaaclab_tasks/manager_based/locomanipulation/pick_place/
├── __init__.py               # 2 个 Gym 任务注册
├── locomanipulation_g1_env_cfg.py
├── fixed_base_upper_body_ik_g1_env_cfg.py
└── configs/

# 模仿学习
isaaclab_mimic/locomanipulation_sdg/envs/g1_locomanipulation_sdg_env.py
isaaclab_mimic/envs/pinocchio_envs/locomanipulation_g1_mimic_env.py

# VR 遥操作
isaaclab/devices/openxr/retargeters/humanoid/unitree/
├── g1_lower_body_standing.py
├── g1_motion_controller_locomotion.py
├── trihand/    (3指灵巧手)
└── inspire/    (5指灵巧手)
```

---

## 八、统计总览

| 类别 | 数量 |
|------|------|
| 运动控制场景 | 4 |
| 操作场景 | 1 |
| 移动操作场景 | 2 |
| 模仿学习场景 | 2 |
| VR 遥操作配置组 | 3 |
| **Gym 任务总计** | **9** |
| VR 遥操作配置组 | **3** |
| 机器人模型配置 | **4** |
