# G1 Inspire 键盘遥操作 — 本 Session 总结（Handoff）

> 用于切换新对话时快速恢复上下文。工作目录：`E:\unitreeG1\unitree_G1\unitree_sim_isaaclab`

---

## 1. 目标

在 **与 RL 训练相同的红块/圆柱抓取场景** 中，实现 **仅右臂** 的键盘遥操作：

- 手腕：Pink IK（与 Isaac Lab 官方 G1 Inspire PickPlace 同路线）
- 手指：直接关节目标 + 渐进闭合（绕过 Pink 24 维 hand 通道对 ring/pinky 驱动不足的问题）
- 左臂 + 左手：全程锁定在 reset 姿态

---

## 2. Conda 环境（必做）

| 项 | 值 |
|----|-----|
| Conda 环境名 | `env_isaaclab` |
| Isaac Sim | `E:\Issac_sim\isaac-sim-standalone-5.1.0-windows-x86_64` |
| Isaac Lab | `E:\Issac_sim\IsaacLab` |
| Pinocchio | 通过 **conda-forge** 安装（不要用 pip 装到 Isaac 自带 Python） |

### 进入环境并启动遥操作

```powershell
conda activate env_isaaclab
cd E:\unitreeG1\unitree_G1\unitree_sim_isaaclab
.\run_teleop.bat
```

`run_teleop.bat` 内部调用：

```bat
E:\Issac_sim\IsaacLab\isaaclab.bat -p teleop_grasp.py --device cuda:0
```

### 注意事项

- **不要与训练同时开**：Isaac Sim 占 GPU，训练 + teleop 二选一
- 启动前先 **点击 Isaac Sim 窗口** 再按键，否则键盘无响应
- **不要按 L**（Se3Keyboard 内置 reset）
- `run_teleop.bat` 必须是 **CRLF + ASCII**（曾出现 LF 导致 cmd 解析错误）

---

## 3. 按键说明

| 键 | 功能 |
|----|------|
| W/S A/D Q/E | 平移右腕 |
| Z/X T/G C/V | 旋转右腕 |
| **K** | 渐进四指抓取（约 1s 闭合） |
| **J** | 更紧抓取 |
| **N** | 张开 |
| **R** | 重置场景 |

**行为约定：**

- 进场景后 **不按移动键 → 全身 idle**（Pink IK 被跳过，关节级锁定）
- **第一次按 WASD/QE 后** → Pink IK 接管右臂；松键 **保持当前位置**（不会弹回）
- 按 **R** → 回到初始姿态

控制台：`[GRASP OK]` / `[hand] idx/mid/ring/pinky=...` 表示抓取检测与四指闭合度。

---

## 4. 本 Session 新增/修改的文件

| 文件 | 作用 |
|------|------|
| `teleop_grasp.py` | 主循环：键盘、Pink 补丁、左/右臂锁定、抓取 UI |
| `teleop_pink_env_cfg.py` | Teleop 环境：仅右腕 Pink IK、圆柱物体、actuator 拆分 |
| `teleop_hand.py` | Pink 24 维 hand 向量（顺序与官方 pickplace 一致） |
| `teleop_fingers.py` | 右手 12 关节直接 PD + 渐进 closure |
| `teleop_grasp_detect.py` | 抓取检测（复用 `env_cfg.py` 奖励逻辑） |
| `run_teleop.bat` | Windows 启动脚本 |

训练相关（同仓库，非 teleop 专用）：`env_cfg.py`、`coupled_grasp_action.py`、`train.py`

---

## 5. 架构要点（新 Session 必读）

### 5.1 Pink IK 与关节锁定的冲突

`env.step()` 内 **decimation=6**，每步 Pink 的 `apply_action()` 会跑 **6 次**，重写右臂 + 24 维 hand 目标。

因此 **仅在 step 外** 调用 `set_joint_position_target()` **无效**。

**当前做法**（`teleop_grasp.py` 内 `_patch_pink_skip_when_idle()`）：

```text
idle（未按移动键）: 跳过 Pink apply → 仅 _enforce_idle_robot()
teleop 中:           orig_apply() → 立刻 _enforce_left_side()
```

### 5.2 Actuator 拆分（`teleop_pink_env_cfg.py`）

| Actuator | 关节 |
|----------|------|
| `left_arm_lock` | 左臂 7 关节，stiffness 10000 |
| `left_hand_lock` | 左手 12 关节，stiffness 10000 |
| `right_arm` | 右臂 7 关节，FTP 官方 PD |
| `hands` | **仅右手** 12 关节 |

### 5.3 物体（teleop 专用）

- 红色 **圆柱**：直径 **4 cm**（与训练方块同宽），高 5.5 cm
- 质量 0.16 kg，高摩擦 + 阻尼，减少一碰就飞

### 5.4 手指抓取

- Pink action 中 **左手始终用 reset 快照角度**
- 右手：K/J 时 `compute_finger_targets()` + 可选 `_finger_ctrl.apply()` 后处理
- Inspire USD 上 **ring/pinky 行程约 0.5 rad**，视觉弯曲比 index/middle 小，属正常

---

## 6. 本 Session 踩坑与修复记录

| 现象 | 原因 | 修复 |
|------|------|------|
| 一进场景右臂自己举起来 | Pink IK + NullSpacePostureTask 每帧覆盖关节目标 | idle 时跳过 Pink；去掉 NullSpacePostureTask |
| 按 K 没反应 | 曾每帧发 `hand_vector_open()` 盖掉闭合 | hand action 写入 closure；`gripper_term=False` + K 回调 |
| 只闭合 index+thumb | Pink 24 维 hand 对 unitree USD 驱动不全 | `teleop_fingers.py` 直接控 12 个右手关节 |
| 圆柱一碰就倒 | 圆柱过粗(7cm)、冲量过大 | 改为 4 cm + 软闭合 ramp + 物体物理调稳 |
| 按移动键左臂也在动 | Pink 每 substep 写左手 24 维目标 | apply 后立刻 `_enforce_left_side()` + `left_hand_lock` |

---

## 7. 参考（GitHub / 官方）

- [Isaac Lab PR #3242](https://github.com/isaac-sim/IsaacLab/pull/3242) — G1 Inspire teleop + DexPilot
- [pickplace_unitree_g1_inspire_hand_env_cfg.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_place/pickplace_unitree_g1_inspire_hand_env_cfg.py)
- [Teleop 文档](https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html)
- 本仓库：`G1_GRASP_NOTES.md`（Pinocchio、WSL、训练踩坑）

---

## 8. 仍未完全解决 / 下一 Session 可继续

1. **四指抓取稳定性**：ring/pinky 闭合度与圆柱包裹仍不理想，可试 DexPilot 式 6  proximal 驱动或 ContactSensor 抓取判定
2. **RL checkpoint 与 teleop 场景对齐**：训练为 4 cm 方块，teleop 为圆柱，策略迁移需单独验证
3. **真机 / XR teleop**：官方路径为 handtracking + `UnitreeG1Retargeter`，非键盘

---

## 9. 新 Session 建议第一句话

> 请读 `unitree_sim_isaaclab/TELEOP_SESSION.md`，继续优化 G1 Inspire 键盘遥操作（`teleop_grasp.py`），环境是 conda `env_isaaclab` + `run_teleop.bat`。
