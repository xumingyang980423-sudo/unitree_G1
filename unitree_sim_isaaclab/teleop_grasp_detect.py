"""Grasp detection for teleop (same criteria as RL env rewards)."""
from __future__ import annotations

import torch

from env_cfg import (
    FOUR_FINGER_PROXIMAL_NAMES,
    _RIGHT_FINGER_CLOSE_RANGE,
    _four_finger_closure,
    _object_to_hand_vec,
    _right_finger_closure,
    _thumb_closure,
)
from teleop_pink_env_cfg import TELEOP_OBJECT_REST_Z


def grasp_metrics(env) -> dict[str, float]:
    robot = env.scene["robot"]
    obj = env.scene["object"]
    dist = float(torch.norm(_object_to_hand_vec(env), dim=1)[0].item())
    four = float(_four_finger_closure(env)[0].item())
    thumb = float(_thumb_closure(env)[0].item())
    all_f = float(_right_finger_closure(env)[0].item())
    obj_z = float(obj.data.root_pos_w[0, 2].item())
    obj_vz = float(obj.data.root_lin_vel_w[0, 2].item())

    per_finger: dict[str, float] = {}
    for name in FOUR_FINGER_PROXIMAL_NAMES:
        lo, hi = _RIGHT_FINGER_CLOSE_RANGE[name]
        pos = float(robot.data.joint_pos[0, robot.data.joint_names.index(name)].item())
        per_finger[name] = (pos - lo) / (hi - lo + 1e-6)

    return {
        "hand_object_dist_m": dist,
        "four_finger_closure": four,
        "thumb_closure": thumb,
        "all_finger_closure": all_f,
        "object_z": obj_z,
        "object_vz": obj_vz,
        "per_finger": per_finger,
    }


def is_grasped(env, min_four: float = 0.30, min_thumb: float = 0.25, max_dist: float = 0.10) -> bool:
    m = grasp_metrics(env)
    return (
        m["four_finger_closure"] >= min_four
        and m["thumb_closure"] >= min_thumb
        and m["hand_object_dist_m"] <= max_dist
    )


def is_lift_grasped(env, lift_delta: float = 0.03, **grasp_kw) -> bool:
    m = grasp_metrics(env)
    if not is_grasped(env, **grasp_kw):
        return False
    return m["object_z"] >= TELEOP_OBJECT_REST_Z + lift_delta
