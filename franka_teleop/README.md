# Franka + G1 键盘遥控 & 演示录制

基于 Isaac Sim 5.1.0 + Isaac Lab，通过键盘遥控 Franka/G1 机器人完成抓取任务，支持录制演示数据用于模仿学习训练。

---

## 环境要求

- Windows 10/11
- Isaac Sim 5.1.0（`E:\Issac_sim\isaac-sim-standalone-5.1.0-windows-x86_64`）
- Isaac Lab 源码（`E:\Issac_sim\IsaacLab`）
- 已执行 `isaaclab.bat -i` 安装扩展包

## 脚本说明

| 脚本 | 功能 |
|------|------|
| `test_teleop_franka.bat` | Franka 键盘遥控，不记录 |
| `test_teleop_g1.bat` | **G1** 键盘遥控（五指手），不记录 |
| `record_demos_franka.bat` | 录制 Franka 演示到 `franka_teleop/datasets/` |
| `replay_demos_franka.bat` | 回放 Franka 演示 |

---

## 1. Franka 键盘遥控

```powershell
cd E:\unitreeG1\unitree_G1\franka_teleop\scripts
.\test_teleop_franka.bat
```

| 按键 | 功能 |
|------|------|
| **W / S** | 末端前后移动（X轴） |
| **A / D** | 末端左右移动（Y轴） |
| **Q / E** | 末端上下移动（Z轴） |
| **Z / X** | 绕 X 轴旋转 |
| **T / G** | 绕 Y 轴旋转 |
| **C / V** | 绕 Z 轴旋转 |
| **K** | 夹爪 张开/闭合 |
| **R** | 重置场景 |

> 确保 Isaac Sim 窗口在前台，键盘事件才会生效。

---

## 2. Franka 录制演示

```powershell
.\record_demos_franka.bat
```

1. 操控 Franka 靠近方块 → 按 K 夹取 → 移到绿色目标球 → 按 K 释放
2. 按 **R** 可丢弃当前演示重来
3. 默认录 10 条，数据存 `franka_teleop/datasets/franka_demos.hdf5`

---

## 3. Franka 回放演示

```powershell
.\replay_demos_franka.bat
```

---

## 4. G1 键盘遥控（五指灵巧手）

```powershell
.\test_teleop_g1.bat
```

G1 使用 PINK IK 控制器，键盘输出右手腕**绝对位姿增量**。左手保持静止，五指同步开/合。

| 按键 | 功能 |
|------|------|
| **W / S** | 右手腕前后移动（X轴） |
| **A / D** | 右手腕左右移动（Y轴） |
| **Q / E** | 右手腕上下移动（Z轴） |
| **Z / X** | 绕 X 轴旋转 |
| **T / G** | 绕 Y 轴旋转 |
| **C / V** | 绕 Z 轴旋转 |
| **K** | 右手五指 张开/闭合 |
| **R** | 重置场景 |

- **机器人**：Unitree G1 + Inspire 五指灵巧手
- **场景**：桌面 + 方向盘物体

---

## 常见问题

**Q: 机器人不动？**
- 确保 Isaac Sim 窗口在前台
- 检查 caps lock
- 点击 Isaac Sim 窗口后再操作

**Q: 夹爪/手指不响应？**
- 切换式的，按 K 切换开/合
- 手臂需靠近物体才能有效抓取

**Q: 崩溃？**
- 检查显卡驱动
- 尝试 `--device cpu`
- 查看 `kit\logs\` 日志
