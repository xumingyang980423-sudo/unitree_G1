# G1 Inspire 抓取抬起 — 方案踩坑记录 & WSL 遥操作指南

本文档记录 **G1 + Inspire 五指手 + 红块抓取抬起** 项目在方案选型上的踩坑（不含 PPO 奖励/超参等训练细节）。  
**当前主路线：Windows 原生 Isaac Lab + PPO 强化学习**（`run_train.bat` / `train.py`）。

---

## 1. 方案路线对比（结论先行）

| 路线 | 平台 | 依赖 | 结论 |
|------|------|------|------|
| **PPO 关节控制 RL** | Windows | 无 Pinocchio | **当前采用**：`env_cfg.py` + `train.py` |
| **PINK IK 键盘遥操作 / 模仿学习** | Linux / WSL | Pinocchio + pin-pink | 官方 G1 操控路线，见下文 WSL 章节 |
| Windows 关节键盘映射 | Windows | 无 Pinocchio | **已废弃**：手感差、手指 bug、与 RL 动作空间不一致 |
| Windows 差分 IK 键盘 | Windows | 无 Pinocchio | **已废弃**：G1 全人形 Jacobian 过重，3060 Laptop 严重卡顿 |
| Windows `pip install pin` | Windows | Pinocchio | **不可行**：PyPI 无 Windows wheel，源码编译卡住 |

**Franka 键盘能精准操控的原因**：`Isaac-Lift-Cube-Franka-IK-Rel-v0` 使用 Isaac Lab 内置 **Differential IK**（单臂 7 DOF、小场景），**不需要 Pinocchio**。G1 官方 pick-place 任务使用 **PINK IK**，与 Franka 不是同一套控制器。

---

## 2. 方案踩坑记录

### 2.1 Windows 无法安装 Pinocchio

- **现象**：`ImportError: DLL load failed while importing pinocchio_pywrap_default`
- **原因**：Isaac Lab 的 `pin-pink` 依赖仅在 **Linux** 写入 `setup.py`；PyPI 的 `pin` 包 **没有 Windows 预编译 wheel**，`pip install pin` 会卡在 *Installing build dependencies*。
- **误判**：在 `AppLauncher` 之前 `import pinocchio` 会同时触发 Omniverse 导入顺序警告，但根因仍是 **Windows 无可用 Pinocchio 二进制**。
- **结论**：在 Windows 上不要尝试 PINK IK / `Isaac-PickPlace-G1-InspireFTP-Abs-v0` 键盘遥操作；改 WSL/Linux 或坚持 PPO RL。

### 2.2 G1 官方任务未自动注册

- **现象**：`NameNotFound: Environment Isaac-PickPlace-G1-InspireFTP-Abs doesn't exist`
- **原因**：`isaaclab_tasks/__init__.py` 为兼容 pinocchio **blacklist 了 `pick_place` 包**，不会自动 import。
- **修复**（Linux/WSL 脚本中）：在 `gym.make` 前显式执行  
  `import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401`

### 2.3 误走「Windows 键盘遥操作」多条死路

#### A. 关节增量映射（RelativeJointPosition）

- 把 Se3 键盘映射到 19 维关节增量。
- **坑 1**：手指每帧写死 `-0.8` 相对动作 → 不按键时手也会持续卷曲，像「自动伸向物块」。
- **坑 2**：关节空间与任务空间不对应，W/S 不等于「手腕前进」，远不如 Franka IK 直观。
- **结论**：已删除相关脚本。

#### B. 差分 IK（仿 Franka DifferentialInverseKinematicsAction）

- **坑 1**：G1 29+ DOF 全人形每帧算 Jacobian，RTX 3060 Laptop + Isaac Sim GUI **极度卡顿**。
- **坑 2**：`teleop_env_cfg.py` 中 `events = None` → 仿真 PLAY 后 `EventManager` 报  
  `'NoneType' object has no attribute '__dict__'`。
- **结论**：已删除；Franka 能跑不代表 G1 能跑。

#### C. `.bat` 编码

- **现象**：`'tlocal'`、`'cho'`、`'M'` 不是内部命令。
- **原因**：批处理中含 Unicode 字符（如 `—`），`cmd.exe` 解析乱码。
- **规则**：`.bat` 只用 **ASCII**，换行 **CRLF**。

### 2.4 PPO 场景 vs 官方 Pick-Place 场景

| 项目 | PPO（`env_cfg.py`） | 官方 IK（`Isaac-PickPlace-G1-InspireFTP-Abs-v0`） |
|------|---------------------|--------------------------------------------------|
| 控制 | 19 维关节增量 | 38 维 PINK IK（双腕 + 双手） |
| 机器人位姿 | `(-4.2, -3.7)` 自定义桌面 | Isaac 官方 G1 桌面布局 |
| 物块 | 6 cm 红立方体 | 任务自带物体 |
| 训练 | skrl PPO | 遥操作 / 模仿学习 |

两套场景 **坐标系不同**，策略与 demo **不能混用**；RL 训练统一用 PPO 场景即可。

### 2.5 模仿学习（IL）在 Windows 上搁置

- 原计划：键盘 → HDF5 → Robomimic BC-RNN（Franka 流程）。
- 阻塞：IL 依赖 PINK IK → 依赖 Pinocchio → Windows 不可用。
- **结论**：已移除 `il/`、`record_demos_g1.*`、`train_bc_g1.*` 等 Windows IL 代码；待 WSL 就绪后再用 **官方任务 + Isaac Lab 自带 `teleop_se3_agent.py`**。

### 2.6 其他易混点

- **`--enable_pinocchio`**：在 Isaac Lab 中主要用于 AppLauncher 的 `pxr.Gf.Matrix4d` patch，**不等于**能在 Windows 装上 Pinocchio。
- **多 env 同步动作**：PPO 早期 16 env 共用同一策略、同一 reset，看起来像「复制人」，属正常探索阶段现象，非 bug。
- **显存**：官方建议 GUI + 渲染 **16GB+ VRAM**；3060 Laptop 6GB 即使用 WSL 也会偏紧，需 `--headless` 或降画质。

---

## 3. 当前推荐：Windows PPO 强化学习

```bat
cd E:\unitreeG1\unitree_G1\unitree_sim_isaaclab
run_train.bat
```

| 脚本 | 用途 |
|------|------|
| `run_train.bat` | PPO 训练（实验名 `g1_grasp_lift_v3`） |
| `run_play.bat` | 加载 checkpoint 演示 |
| `view_scene.bat` | 仅查看 G1 + 红块场景 |
| `run_train_demo.bat` | 单 env 快速试跑 |

核心配置：`env_cfg.py`（奖励门控抓取、分指缩放、腕部距离惩罚等）。

---

## 4. WSL 方案：能否看到 GUI？

**可以。** Windows 11 自带 **WSLg**，WSL 内启动的 Isaac Sim 会以 **普通 Windows 窗口** 显示 3D 视口，无需 VcXsrv。

前提：

```bash
# Windows PowerShell
wsl --version          # 需 WSL2 + WSLg

# WSL Ubuntu 内
nvidia-smi             # 需能看到 RTX 3060（GPU 透传）
echo $DISPLAY          # 通常有值，表示 WSLg 就绪
```

注意：WSLg 性能略低于原生 Windows Isaac Sim；6GB 显存仍偏紧。

---

## 5. WSL 安装清单（Ubuntu 22.04 + Isaac Lab + Pinocchio + G1 PINK IK）

### 5.1 Windows 侧准备

```powershell
# 1. 启用 WSL2 + Ubuntu 22.04（Microsoft Store 或 wsl --install -d Ubuntu-22.04）
wsl --update
wsl --shutdown

# 2. 确保 NVIDIA 驱动为 Windows 侧最新（支持 WSL CUDA）
# 3. 进入 WSL
wsl -d Ubuntu-22.04
```

### 5.2 WSL 内基础依赖

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git wget curl unzip \
    libgl1-mesa-dev libglu1-mesa-dev libxt-dev libx11-dev

# 验证 GPU
nvidia-smi
```

### 5.3 安装 Isaac Sim（Linux 二进制）

按 [Isaac Lab 官方文档 — Binaries Installation (Linux)](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html)：

```bash
# 建议放在 Linux 家目录，避免 /mnt/e 性能差
mkdir -p ~/isaac && cd ~/isaac

# 从 NVIDIA 下载 Isaac Sim 5.1 Linux 压缩包并解压，例如：
# tar -xf isaac-sim-5.1.0-linux-x86_64.zip
export ISAACSIM_PATH=~/isaac/isaac-sim-5.1.0
export ISAACSIM_PYTHON_EXE=$ISAACSIM_PATH/python.sh

# 接受 EULA
$ISAACSIM_PYTHON_EXE -m pip install --upgrade pip
```

### 5.4 克隆并安装 Isaac Lab

```bash
cd ~
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
# 与 Windows 上版本对齐（例如 2.x / 对应 Sim 5.1 的分支或 tag）
git checkout main   # 或你 Windows 上正在用的 tag

# 链接 Isaac Sim（二选一）
ln -s $ISAACSIM_PATH _isaac_sim

# 安装 Isaac Lab + RL 依赖 + Linux 版 pin-pink
./isaaclab.sh -i
```

Linux 下 `./isaaclab.sh -i` 会安装 **`pin-pink` → Pinocchio**，这是 PINK IK 的关键。

若 Pinocchio 仍有问题，可备用：

```bash
conda install -c conda-forge pinocchio
# 或
$ISAACSIM_PYTHON_EXE -m pip install pin
```

### 5.5 挂载 Windows 工程（可选）

```bash
# 只读/开发用；训练数据建议复制到 ~/ 下
ls /mnt/e/unitreeG1/unitree_G1/unitree_sim_isaaclab
```

### 5.6 验证 Pinocchio + 官方 G1 任务

```bash
cd ~/IsaacLab
export OMNI_KIT_ACCEPT_EULA=YES

# 快速 import 测试
./isaaclab.sh -p -c "import pinocchio; print('pinocchio', pinocchio.__version__)"

# 列出任务（需先 import pick_place）
./isaaclab.sh -p -c "
import isaaclab_tasks
import isaaclab_tasks.manager_based.manipulation.pick_place
import gymnasium as gym
print(gym.spec('Isaac-PickPlace-G1-InspireFTP-Abs-v0'))
"
```

### 5.7 G1 PINK IK 键盘遥操作（官方脚本）

使用 Isaac Lab 自带 `teleop_se3_agent.py`（与 Franka 同源，需 `--enable_pinocchio`）：

```bash
cd ~/IsaacLab
export OMNI_KIT_ACCEPT_EULA=YES

./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
    --task Isaac-PickPlace-G1-InspireFTP-Abs-v0 \
    --teleop_device keyboard \
    --num_envs 1 \
    --device cuda:0 \
    --enable_pinocchio
```

**按键**（与 Franka 相同）：

| 按键 | 功能 |
|------|------|
| W/S | X 轴平移 |
| A/D | Y 轴平移 |
| Q/E | Z 轴平移 |
| Z/X T/G C/V | 旋转 |
| K | 夹爪/手开合并 |
| R | 重置（若脚本绑定） |

**操作要点**：启动后 **用鼠标点击 Isaac Sim 3D 视口**，再按键。

### 5.8 录制演示 + 模仿学习（WSL，可选后续）

官方工具链（Linux + Pinocchio）：

```bash
# 录制
./isaaclab.sh -p scripts/tools/record_demos.py \
    --task Isaac-PickPlace-G1-InspireFTP-Abs-v0 \
    --teleop_device keyboard \
    --dataset_file ./datasets/g1_pickplace_demos.hdf5 \
    --num_demos 20 \
    --device cuda:0 \
    --enable_pinocchio

# 回放
./isaaclab.sh -p scripts/tools/replay_demos.py \
    --dataset_file ./datasets/g1_pickplace_demos.hdf5 \
    --enable_pinocchio

# Robomimic 训练（见 Isaac Lab teleop_imitation 文档）
./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
    --task Isaac-PickPlace-G1-InspireFTP-Abs-v0 \
    --dataset ./datasets/g1_pickplace_demos.hdf5 \
    --enable_pinocchio
```

### 5.9 WSL 常见问题

| 问题 | 处理 |
|------|------|
| `nvidia-smi` 在 WSL 不可用 | 升级 Windows NVIDIA 驱动，重启，`wsl --shutdown` |
| GUI 黑屏 | `wsl --update`；确认不是 SSH 远程会话 |
| 很卡 / OOM | `--headless` 录数据；减小 `--num_envs`；关 RTX 其他占用 |
| `/mnt/e` 极慢 | 把 Isaac Sim / 数据集放到 `~/isaac` |
| Pinocchio import 失败 | 重新 `./isaaclab.sh -i`；检查 `import pinocchio` 在 **AppLauncher 之后** |

---

## 6. 仓库内相关目录（清理后）

```text
unitree_sim_isaaclab/
  env_cfg.py          # PPO 环境（主路线）
  train.py / play.py
  run_train.bat       # Windows 训练入口
  view_scene.py       # 场景预览
  G1_GRASP_NOTES.md   # 本文档

franka_teleop/        # Franka / G1 键盘参考（需在 Linux/WSL + Pinocchio 下跑 G1）
```

---

## 7. 决策树（速查）

```
要在 Windows 上抓红块抬 5cm？
  └─> run_train.bat（PPO RL）✓

要键盘精准遥操作 / 录 IL demo？
  └─> WSL Ubuntu 22.04 + Pinocchio
        └─> teleop_se3_agent.py + Isaac-PickPlace-G1-InspireFTP-Abs-v0 ✓

能在 Windows 上不装 Pinocchio 达到 Franka 级 IK 手感？
  └─> 基本不行（已验证关节映射、差分 IK 两条路）✗
```

---

*文档版本：2026-05，对应当前 `unitree_sim_isaaclab` PPO v3 训练配置。*
