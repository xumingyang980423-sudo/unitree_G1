"""Pink IK hand targets (24-D), matching pick_place hand_joint_names order."""
from __future__ import annotations

import numpy as np

from coupled_grasp_action import (
    FOUR_FINGER_JOINT_NAMES,
    THUMB_JOINT_NAMES,
    _FOUR_FINGER_SPEC,
    _THUMB_SPEC,
)
from teleop_fingers import INSPIRE_DRIVE_JOINTS, compute_drive_targets, compute_finger_targets

# Same order as pickplace_unitree_g1_inspire_hand_env_cfg.ActionsCfg.pink_ik_cfg.hand_joint_names
PINK_HAND_JOINT_NAMES: tuple[str, ...] = (
    "L_index_proximal_joint",
    "L_middle_proximal_joint",
    "L_pinky_proximal_joint",
    "L_ring_proximal_joint",
    "L_thumb_proximal_yaw_joint",
    "R_index_proximal_joint",
    "R_middle_proximal_joint",
    "R_pinky_proximal_joint",
    "R_ring_proximal_joint",
    "R_thumb_proximal_yaw_joint",
    "L_index_intermediate_joint",
    "L_middle_intermediate_joint",
    "L_pinky_intermediate_joint",
    "L_ring_intermediate_joint",
    "L_thumb_proximal_pitch_joint",
    "R_index_intermediate_joint",
    "R_middle_intermediate_joint",
    "R_pinky_intermediate_joint",
    "R_ring_intermediate_joint",
    "R_thumb_proximal_pitch_joint",
    "L_thumb_intermediate_joint",
    "R_thumb_intermediate_joint",
    "L_thumb_distal_joint",
    "R_thumb_distal_joint",
)

# Right-hand Pink action: 6 Inspire drive joints only (intermediate/distal are mimics in USD).
PINK_RIGHT_HAND_DRIVE_JOINT_NAMES: tuple[str, ...] = INSPIRE_DRIVE_JOINTS

# Legacy 12-D order kept for reference / open-pose helpers.
PINK_RIGHT_HAND_JOINT_NAMES: tuple[str, ...] = tuple(n for n in PINK_HAND_JOINT_NAMES if n.startswith("R_"))

_OPEN_VALUE = 0.05
# Partial closure for cylinder teleop (full max causes finger-finger interpenetration).
_FINGER_GRASP_U = 0.70
_THUMB_GRASP_U = 0.75


def _open_value(name: str) -> float:
    spec = _FOUR_FINGER_SPEC.get(name) or _THUMB_SPEC.get(name)
    if spec is None:
        return _OPEN_VALUE
    lo, _, _ = spec
    return max(lo, _OPEN_VALUE)


def _grasp_value(name: str, closure_u: float) -> float:
    """Absolute joint target using the same per-joint roles as RL coupled_grasp_action."""
    spec = _FOUR_FINGER_SPEC.get(name) or _THUMB_SPEC.get(name)
    if spec is None:
        return _open_value(name)
    lo, hi, role = spec
    span = hi - lo
    return float(lo + closure_u * role * span)


def _dict_from_vector(values: np.ndarray) -> dict[str, float]:
    return {name: float(values[i]) for i, name in enumerate(PINK_HAND_JOINT_NAMES)}


def _vector_from_dict(joints: dict[str, float]) -> np.ndarray:
    return np.array([joints[name] for name in PINK_HAND_JOINT_NAMES], dtype=np.float32)


def hand_vector_open() -> np.ndarray:
    return np.array([_open_value(n) for n in PINK_HAND_JOINT_NAMES], dtype=np.float32)


def hand_vector_grasp_right() -> np.ndarray:
    """Cylindrical grasp: four fingers curl together + opposing thumb (not joint-limit crush)."""
    joints = _dict_from_vector(hand_vector_open())
    for name in FOUR_FINGER_JOINT_NAMES:
        joints[name] = _grasp_value(name, _FINGER_GRASP_U)
    for name in THUMB_JOINT_NAMES:
        joints[name] = _grasp_value(name, _THUMB_GRASP_U)
    return _vector_from_dict(joints)


def hand_vector_from_named_positions(named: dict[str, float]) -> np.ndarray:
    """Build 24-D Pink hand vector from per-joint positions (missing names keep open pose)."""
    joints = _dict_from_vector(hand_vector_open())
    joints.update(named)
    return _vector_from_dict(joints)


def right_hand_vector_from_named_positions(named: dict[str, float]) -> np.ndarray:
    """Build 6-D Pink hand vector (Inspire drive joints, DDS order)."""
    drives = compute_drive_targets(0.0)
    drives.update({k: v for k, v in named.items() if k in INSPIRE_DRIVE_JOINTS})
    return np.array([drives[name] for name in INSPIRE_DRIVE_JOINTS], dtype=np.float32)


def hand_vector_for_closure(closure: float, tight: bool = False) -> np.ndarray:
    """24-D Pink hand action: left hand open, right hand at `closure` in [0, 1]."""
    if closure <= 0.0:
        return hand_vector_open()
    targets = compute_finger_targets(closure, tight=tight)
    joints = _dict_from_vector(hand_vector_open())
    for name in FOUR_FINGER_JOINT_NAMES + THUMB_JOINT_NAMES:
        joints[name] = targets[name]
    return _vector_from_dict(joints)
