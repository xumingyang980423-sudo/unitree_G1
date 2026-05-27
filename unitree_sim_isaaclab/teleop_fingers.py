"""Direct right-hand joint control for teleop (Inspire 6-drive + explicit mimic targets)."""
from __future__ import annotations

import torch

from coupled_grasp_action import (
    FOUR_FINGER_JOINT_NAMES,
    THUMB_JOINT_NAMES,
    _FOUR_FINGER_SPEC,
    _THUMB_SPEC,
)

RIGHT_HAND_JOINTS = FOUR_FINGER_JOINT_NAMES + THUMB_JOINT_NAMES

INSPIRE_DRIVE_JOINTS: tuple[str, ...] = (
    "R_pinky_proximal_joint",
    "R_ring_proximal_joint",
    "R_middle_proximal_joint",
    "R_index_proximal_joint",
    "R_thumb_proximal_pitch_joint",
    "R_thumb_proximal_yaw_joint",
)

_INSPIRE_MIMIC: dict[str, tuple[str, float]] = {
    "R_index_intermediate_joint": ("R_index_proximal_joint", 1.0),
    "R_middle_intermediate_joint": ("R_middle_proximal_joint", 1.0),
    "R_pinky_intermediate_joint": ("R_pinky_proximal_joint", 1.0),
    "R_ring_intermediate_joint": ("R_ring_proximal_joint", 1.0),
    "R_thumb_intermediate_joint": ("R_thumb_proximal_pitch_joint", 1.5),
    "R_thumb_distal_joint": ("R_thumb_proximal_pitch_joint", 2.4),
}

_FINGER_PROXIMAL_SCALE: dict[str, float] = {
    "R_index_proximal_joint": 0.42,
    "R_middle_proximal_joint": 0.42,
    "R_ring_proximal_joint": 1.00,
    "R_pinky_proximal_joint": 1.00,
}

_THUMB_YAW_JOINT = "R_thumb_proximal_yaw_joint"
_THUMB_PITCH_JOINT = "R_thumb_proximal_pitch_joint"

# Side grasp: yaw to far side; moderate pitch aligns pad with cylinder mid-height (not top hook).
_THUMB_YAW_OPEN = 0.06
_THUMB_YAW_SIDE = 0.54
_THUMB_YAW_CLOSE = 0.64
_THUMB_PITCH_OPEN = 0.04
_THUMB_PITCH_SIDE = 0.28
_THUMB_PITCH_CLOSE = 0.34
_THUMB_PITCH_TIGHT_EXTRA = 0.04

# Do not curl four fingers / thumb pinch until palm is this close to the object (meters).
_GRASP_CLOSE_MAX_DIST = 0.095
_GRASP_FULL_CLOSE_DIST = 0.075

_RING_PINKY_U_BOOST = 1.10
_RING_PINKY_TIGHT_U_BOOST = 1.15

_THUMB_PREP_STEP = 0.008
_PINCH_STEP = 0.012
_FINGER_STEP = 0.014
_OPEN_STEP = 0.022
_THUMB_PREP_NEAR_SCALE = 0.40


def _joint_limits(name: str) -> tuple[float, float]:
    spec = _FOUR_FINGER_SPEC.get(name) or _THUMB_SPEC.get(name)
    if spec is None:
        return 0.0, 1.0
    lo, hi, _ = spec
    return lo, hi


def _clamp(name: str, value: float) -> float:
    lo, hi = _joint_limits(name)
    return float(max(lo, min(hi, value)))


def _lerp(a: float, b: float, t: float) -> float:
    t = max(0.0, min(1.0, t))
    return a + t * (b - a)


def _thumb_drive_u(joint_name: str, closure_u: float) -> float:
    lo, hi, role = _THUMB_SPEC[joint_name]
    return float(lo + closure_u * role * (hi - lo))


def _close_scale_from_distance(dist_m: float) -> float:
    if dist_m >= _GRASP_CLOSE_MAX_DIST:
        return 0.0
    if dist_m <= _GRASP_FULL_CLOSE_DIST:
        return 1.0
    t = (_GRASP_CLOSE_MAX_DIST - dist_m) / (_GRASP_CLOSE_MAX_DIST - _GRASP_FULL_CLOSE_DIST)
    return max(0.0, min(1.0, t))


def _thumb_side_prep_drives(thumb_prep_u: float) -> tuple[float, float]:
    """Move thumb to cylinder SIDE via yaw + mid-height pitch."""
    u = max(0.0, min(1.0, thumb_prep_u))
    yaw_u = _lerp(_THUMB_YAW_OPEN, _THUMB_YAW_SIDE, u)
    pitch_u = _lerp(_THUMB_PITCH_OPEN, _THUMB_PITCH_SIDE, u)
    return _thumb_drive_u(_THUMB_YAW_JOINT, yaw_u), _thumb_drive_u(_THUMB_PITCH_JOINT, pitch_u)


def _thumb_side_pinch_drives(pinch_u: float, tight: bool) -> tuple[float, float]:
    """Close from the side: yaw opposition first; pitch rises only slightly."""
    pu = max(0.0, min(1.0, pinch_u))
    yaw_u = _lerp(_THUMB_YAW_SIDE, _THUMB_YAW_CLOSE, pu)
    pitch_u = _lerp(_THUMB_PITCH_SIDE, _THUMB_PITCH_CLOSE, pu * 0.65)
    if tight:
        pitch_u = min(1.0, pitch_u + _THUMB_PITCH_TIGHT_EXTRA)
    return _thumb_drive_u(_THUMB_YAW_JOINT, yaw_u), _thumb_drive_u(_THUMB_PITCH_JOINT, pitch_u)


def thumb_grasp_phase(thumb_prep_u: float, pinch_u: float) -> str:
    if thumb_prep_u <= 0.0 and pinch_u <= 0.0:
        return "open"
    if thumb_prep_u < 1.0:
        return "side"
    return "close"


def compute_drive_targets_from_state(
    thumb_prep_u: float,
    pinch_u: float,
    finger_u: float,
    tight: bool = False,
) -> dict[str, float]:
    targets: dict[str, float] = {}
    if thumb_prep_u < 1.0:
        thumb_yaw, thumb_pitch = _thumb_side_prep_drives(thumb_prep_u)
    else:
        thumb_yaw, thumb_pitch = _thumb_side_pinch_drives(pinch_u, tight)
    targets[_THUMB_YAW_JOINT] = thumb_yaw
    targets[_THUMB_PITCH_JOINT] = thumb_pitch

    fu = max(0.0, min(1.0, finger_u))
    for name in INSPIRE_DRIVE_JOINTS:
        if name in (_THUMB_YAW_JOINT, _THUMB_PITCH_JOINT):
            continue
        lo, hi, role = _FOUR_FINGER_SPEC[name]
        scale = _FINGER_PROXIMAL_SCALE[name]
        if "ring" in name or "pinky" in name:
            f = min(1.0, fu * _RING_PINKY_U_BOOST)
            if tight:
                f = min(1.0, fu * _RING_PINKY_TIGHT_U_BOOST)
            targets[name] = float(lo + f * scale * role * (hi - lo))
        else:
            targets[name] = float(lo + fu * scale * role * (hi - lo))
    return targets


def compute_drive_targets(closure: float, tight: bool = False) -> dict[str, float]:
    prep = max(0.0, min(1.0, closure))
    rest = max(0.0, min(1.0, closure - 1.0)) if closure > 1.0 else 0.0
    return compute_drive_targets_from_state(prep, rest, rest, tight=tight)


def compute_finger_targets_from_state(
    thumb_prep_u: float,
    pinch_u: float,
    finger_u: float,
    tight: bool = False,
) -> dict[str, float]:
    targets = compute_drive_targets_from_state(thumb_prep_u, pinch_u, finger_u, tight=tight)
    pitch = targets[_THUMB_PITCH_JOINT]
    side_grasp = thumb_prep_u < 1.0 or pinch_u < 0.85
    for mimic_name, (src, mult) in _INSPIRE_MIMIC.items():
        if mimic_name in ("R_thumb_intermediate_joint", "R_thumb_distal_joint") and side_grasp:
            # Low pitch + no distal amplification => pad stays on cylinder side, not over top.
            targets[mimic_name] = _clamp(mimic_name, pitch * (1.0 if "intermediate" in mimic_name else 1.05))
            continue
        targets[mimic_name] = _clamp(mimic_name, targets[src] * mult)
    return targets


def compute_finger_targets(closure: float, tight: bool = False) -> dict[str, float]:
    prep = max(0.0, min(1.0, closure))
    rest = max(0.0, min(1.0, closure - 1.0)) if closure > 1.0 else 0.0
    return compute_finger_targets_from_state(prep, rest, rest, tight=tight)


class DirectFingerController:
    """Thumb side-prep then side-pinch; four fingers close after thumb prep completes."""

    def __init__(self) -> None:
        self.thumb_prep_u = 0.0
        self.pinch_u = 0.0
        self.finger_u = 0.0
        self.tight = False
        self._want_closed = False
        self._near_scale = 1.0
        self._hand_dist_m = 999.0
        self._joint_ids: list[int] | None = None
        self._device: torch.device | None = None

    @property
    def closure(self) -> float:
        if self.thumb_prep_u < 1.0:
            return self.thumb_prep_u
        return 1.0 + 0.5 * (self.pinch_u + self.finger_u)

    @property
    def down_ready(self) -> bool:
        return self.thumb_prep_u >= 1.0

    def reset(self, robot) -> None:
        self.thumb_prep_u = 0.0
        self.pinch_u = 0.0
        self.finger_u = 0.0
        self.tight = False
        self._want_closed = False
        self._hand_dist_m = 999.0
        self._joint_ids = [robot.data.joint_names.index(n) for n in RIGHT_HAND_JOINTS]
        self._device = robot.data.joint_pos.device

    def grasp_phase(self) -> str:
        return thumb_grasp_phase(self.thumb_prep_u, self.pinch_u)

    def in_prep_phase(self) -> bool:
        return self.thumb_prep_u < 1.0

    @property
    def contact_allowed(self) -> bool:
        return _close_scale_from_distance(self._hand_dist_m) > 0.0

    def set_hand_distance(self, dist_m: float) -> None:
        self._hand_dist_m = max(0.0, float(dist_m))

    def set_closed(self, closed: bool, tight: bool = False, near_scale: float = 1.0) -> None:
        self._want_closed = closed
        self.tight = tight
        self._near_scale = max(0.35, min(1.0, near_scale))

    def _advance_state(self) -> None:
        if not self._want_closed:
            self.thumb_prep_u = max(0.0, self.thumb_prep_u - _OPEN_STEP)
            self.pinch_u = max(0.0, self.pinch_u - _OPEN_STEP)
            self.finger_u = max(0.0, self.finger_u - _OPEN_STEP)
            return

        scale = self._near_scale * _close_scale_from_distance(self._hand_dist_m)
        if self.thumb_prep_u < 1.0:
            prep_scale = min(max(scale, 0.25), _THUMB_PREP_NEAR_SCALE) if scale > 0 else min(
                self._near_scale, _THUMB_PREP_NEAR_SCALE
            )
            self.thumb_prep_u = min(1.0, self.thumb_prep_u + _THUMB_PREP_STEP * prep_scale)
            return

        if scale <= 0.0:
            return

        self.pinch_u = min(1.0, self.pinch_u + _PINCH_STEP * scale)
        self.finger_u = min(1.0, self.finger_u + _FINGER_STEP * scale)

    def apply(self, robot) -> None:
        if self._joint_ids is None:
            self.reset(robot)
        self._advance_state()
        if (
            self.thumb_prep_u <= 0.0
            and self.pinch_u <= 0.0
            and self.finger_u <= 0.0
            and not self._want_closed
        ):
            return
        targets = compute_finger_targets_from_state(
            self.thumb_prep_u, self.pinch_u, self.finger_u, tight=self.tight
        )
        values = torch.tensor(
            [targets[n] for n in RIGHT_HAND_JOINTS],
            device=self._device,
            dtype=torch.float32,
        ).unsqueeze(0)
        robot.set_joint_position_target(values, joint_ids=self._joint_ids)
