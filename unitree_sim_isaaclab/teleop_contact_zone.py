"""Target contact zone for teleop side-grasp on the red cylinder.

Side grasp geometry (robot-centric):
- Palm faces the cylinder; four fingers on one side, thumb on the other.
- Best contact is on the **middle of the proximal phalanx** (指根前面、指尖后面),
  not on the palm and not only on the fingertips.

We expose two measurable proxies in sim:
1. dist — hand centroid ↔ object center (depth along approach)
2. height_err — object Z minus finger-pad row Z (vertical alignment)
"""
from __future__ import annotations

import torch

from env_cfg import _right_hand_approach_pos

# Hand centroid ↔ object center (m). Too small = object at finger base/palm; too large = miss.
CONTACT_DIST_MIN_M = 0.028
CONTACT_DIST_IDEAL_M = 0.040
CONTACT_DIST_MAX_M = 0.055

# Object Z vs index/middle pad row Z (m). Positive = object above finger row.
CONTACT_HEIGHT_ERR_MIN_M = -0.018
CONTACT_HEIGHT_ERR_IDEAL_M = 0.0
CONTACT_HEIGHT_ERR_MAX_M = 0.018

_FINGER_PAD_PREFIXES = ("R_index", "R_middle")


def _finger_pad_row_z(env) -> float:
    """Mean Z of index/middle proximal links (pad row height)."""
    robot = env.scene["robot"]
    zs: list[float] = []
    for i, name in enumerate(robot.data.body_names):
        if not any(name.startswith(p) for p in _FINGER_PAD_PREFIXES):
            continue
        if "proximal" not in name.lower():
            continue
        zs.append(float(robot.data.body_pos_w[0, i, 2].item()))
    if not zs:
        return float(_right_hand_approach_pos(env)[0, 2].item())
    return sum(zs) / len(zs)


def _soft_band(value: float, lo: float, ideal: float, hi: float) -> float:
    if value <= lo or value >= hi:
        return 0.0
    if value <= ideal:
        return (value - lo) / (ideal - lo + 1e-9)
    return (hi - value) / (hi - ideal + 1e-9)


def contact_zone_metrics(env) -> dict[str, float | str | bool]:
    obj = env.scene["object"].data.root_pos_w[0]
    hand = _right_hand_approach_pos(env)[0]
    dist = float(torch.norm(obj - hand).item())
    obj_z = float(obj[2].item())
    pad_z = _finger_pad_row_z(env)
    height_err = obj_z - pad_z

    dist_ok = CONTACT_DIST_MIN_M <= dist <= CONTACT_DIST_MAX_M
    height_ok = CONTACT_HEIGHT_ERR_MIN_M <= height_err <= CONTACT_HEIGHT_ERR_MAX_M
    zone_score = _soft_band(dist, CONTACT_DIST_MIN_M, CONTACT_DIST_IDEAL_M, CONTACT_DIST_MAX_M) * _soft_band(
        height_err, CONTACT_HEIGHT_ERR_MIN_M, CONTACT_HEIGHT_ERR_IDEAL_M, CONTACT_HEIGHT_ERR_MAX_M
    )

    if dist < CONTACT_DIST_MIN_M:
        label = "too_close"  # 偏指根/掌心
    elif dist > CONTACT_DIST_MAX_M:
        label = "too_far"
    elif height_err > CONTACT_HEIGHT_ERR_MAX_M:
        label = "object_high"  # 物体偏高，四指够不到同一高度
    elif height_err < CONTACT_HEIGHT_ERR_MIN_M:
        label = "object_low"
    else:
        label = "ideal"

    return {
        "contact_dist_m": dist,
        "contact_dist_ok": dist_ok,
        "contact_height_err_m": height_err,
        "contact_height_ok": height_ok,
        "finger_pad_z": pad_z,
        "object_z": obj_z,
        "contact_zone_score": float(zone_score),
        "contact_zone_label": label,
        "contact_zone_ok": dist_ok and height_ok,
    }


def contact_zone_hint(label: str) -> str:
    hints = {
        "ideal": "位置合适，可以 K 合拢",
        "too_close": "太近指根：WASD 微退 / 略抬高腕部，让圆柱到指节中段",
        "too_far": "太远：靠近一些再 K（dist 目标 28–55 mm）",
        "object_high": "物体偏高：Q 降腕或 S 后退，让圆柱对齐四指横排",
        "object_low": "物体偏低：E 抬腕，让圆柱对齐四指横排（否则 ring/pinky 够不到）",
    }
    return hints.get(label, "")
