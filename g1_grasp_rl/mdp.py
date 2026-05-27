"""Custom MDP functions for G1 grasp RL (joint position control)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    obj: RigidObject = env.scene[asset_cfg.name]
    return torch.where(obj.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def eef_to_object_distance(
    env: ManagerBasedRLEnv,
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    body_name: str = "right_wrist_yaw_link",
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    robot = env.scene[robot_cfg.name]
    ee_idx = robot.data.body_names.index(body_name)
    ee_pos = robot.data.body_pos_w[:, ee_idx]
    dist = torch.norm(obj.data.root_pos_w - ee_pos, dim=1)
    return 1.0 - torch.tanh(dist / std)


def eef_to_object_distance_raw(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    body_name: str = "right_wrist_yaw_link",
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    robot = env.scene[robot_cfg.name]
    ee_idx = robot.data.body_names.index(body_name)
    ee_pos = robot.data.body_pos_w[:, ee_idx]
    return obj.data.root_pos_w - ee_pos
