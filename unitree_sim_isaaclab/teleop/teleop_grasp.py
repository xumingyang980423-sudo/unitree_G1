"""Keyboard teleop: red-block scene + Pink IK (right arm only)."""
from __future__ import annotations

import argparse
import os
import re
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PROJECT_ROOT"] = _repo_root
sys.path.insert(0, _repo_root)

_ISAAC_SIM_PATH = "E:\\Issac_sim\\isaac-sim-standalone-5.1.0-windows-x86_64"
os.add_dll_directory(os.path.join(_ISAAC_SIM_PATH, "kit", "python", "Lib", "site-packages", "h5py"))
import h5py  # noqa: F401

del h5py
os.environ.setdefault("CARB_APP_PATH", os.path.join(_ISAAC_SIM_PATH, "kit"))
os.environ.setdefault("ISAAC_PATH", _ISAAC_SIM_PATH)
os.environ.setdefault("EXP_PATH", os.path.join(_ISAAC_SIM_PATH, "apps"))

_pinocchio_deps = os.path.join(
    _ISAAC_SIM_PATH, "extscache", "omni.usd.libs-1.0.1+69cbf6ad.wx64.r.cp311", "bin"
)
if os.path.isdir(_pinocchio_deps):
    os.add_dll_directory(_pinocchio_deps)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Red-block scene keyboard teleop (Pink IK, right arm).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

import pinocchio  # noqa: F401

args_cli.headless = False
args_cli.num_envs = 1

import isaacsim  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.envs import ManagerBasedRLEnv

from teleop.teleop_fingers import (
    DirectFingerController,
    RIGHT_HAND_JOINTS,
    _GRASP_CLOSE_MAX_DIST,
    compute_drive_targets,
    compute_finger_targets_from_state,
)
from teleop.teleop_grasp_detect import grasp_metrics, is_grasped, is_lift_grasped

from teleop.teleop_contact_zone import (
    CONTACT_DIST_IDEAL_M,
    CONTACT_DIST_MAX_M,
    CONTACT_DIST_MIN_M,
    contact_zone_hint,
    contact_zone_metrics,
)
from teleop.teleop_hand import right_hand_vector_from_named_positions
from teleop.teleop_pink_env_cfg import (
    BODY_LOCK_JOINTS,
    LEG_LOCK_JOINTS_EXPR,
    RIGHT_ARM_JOINTS,
    make_teleop_pink_env_cfg,
)

RIGHT_HAND_JOINT_NAMES = RIGHT_HAND_JOINTS

# Workspace boundary in pelvis-local frame.
# local +Y = robot left, local -Y = robot right.
# Keep the right wrist on the right side without projecting A/D motion outward.
_WS_MAX_LOCAL_Y = -0.14
# In pelvis-local frame: don't reach too far behind the body.
_WS_MIN_LOCAL_X = -0.05
# Height bounds (world Z): don't go below hip or above head.
_WS_MIN_Z = 0.62
_WS_MAX_Z = 1.25

right_pos = np.zeros(3, dtype=np.float64)
right_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
gripper_closed = False
gripper_tight = False
_finger_ctrl = DirectFingerController()
_grasp_state = "open"
_status_counter = 0
_grasp_far_warned = False
_body_lock_joint_ids: list[int] | None = None
_body_lock_hold: torch.Tensor | None = None
_right_arm_joint_ids: list[int] | None = None
_right_arm_hold: torch.Tensor | None = None
_right_hand_joint_ids: list[int] | None = None
_right_hand_hold: torch.Tensor | None = None
_idle_right_pos = np.zeros(3, dtype=np.float64)
_idle_right_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
_arm_moving = False

env_cfg = make_teleop_pink_env_cfg(1, sim_device=args_cli.device)
env = ManagerBasedRLEnv(cfg=env_cfg)
teleop = Se3Keyboard(
    Se3KeyboardCfg(pos_sensitivity=0.03, rot_sensitivity=0.06, gripper_term=False)
)


def _right_wrist_pose() -> tuple[np.ndarray, np.ndarray]:
    robot = env.scene["robot"]
    idx = robot.data.body_names.index("right_wrist_yaw_link")
    pos = robot.data.body_pos_w[0, idx].cpu().numpy().astype(np.float64)
    quat = robot.data.body_quat_w[0, idx].cpu().numpy().astype(np.float64)
    return pos, quat


def _pelvis_pose() -> tuple[np.ndarray, R]:
    """Return pelvis world position and scipy Rotation."""
    robot = env.scene["robot"]
    idx = robot.data.body_names.index("pelvis")
    pos = robot.data.body_pos_w[0, idx].cpu().numpy().astype(np.float64)
    quat = robot.data.body_quat_w[0, idx].cpu().numpy().astype(np.float64)
    rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # xyzw
    return pos, rot


def _clamp_wrist_target(target: np.ndarray) -> np.ndarray:
    """Clamp IK target to the safe workspace envelope relative to the pelvis.

    Works in pelvis-local frame so the bounds are body-relative regardless of
    the robot's world-space position and orientation.
    """
    pelvis_pos, pelvis_rot = _pelvis_pose()

    target[2] = np.clip(target[2], _WS_MIN_Z, _WS_MAX_Z)

    # World -> pelvis local frame (XY plane only for lateral/frontal bounds).
    delta_world = target - pelvis_pos
    delta_local = pelvis_rot.inv().apply(delta_world)

    clamped = False

    # Don't reach behind the body.
    if delta_local[0] < _WS_MIN_LOCAL_X:
        delta_local[0] = _WS_MIN_LOCAL_X
        clamped = True

    # Right arm: stay on the right side (negative local-Y is right, positive is left).
    if delta_local[1] > _WS_MAX_LOCAL_Y:
        delta_local[1] = _WS_MAX_LOCAL_Y
        clamped = True

    if clamped:
        target[:] = pelvis_pos + pelvis_rot.apply(delta_local)

    return target


def _leg_joint_names(joint_names: list[str]) -> list[str]:
    patterns = [re.compile(expr) for expr in LEG_LOCK_JOINTS_EXPR]
    return [n for n in joint_names if any(p.fullmatch(n) for p in patterns)]


def _cache_arm_holds() -> None:
    """Snapshot frozen body (left side, waist, legs) + right arm/hand at reset pose."""
    global _body_lock_joint_ids, _body_lock_hold
    global _right_arm_joint_ids, _right_arm_hold, _right_hand_joint_ids, _right_hand_hold
    global _idle_right_pos, _idle_right_quat
    robot = env.scene["robot"]
    lock_names = list(dict.fromkeys(BODY_LOCK_JOINTS + _leg_joint_names(list(robot.data.joint_names))))
    _body_lock_joint_ids = [robot.data.joint_names.index(n) for n in lock_names]
    _body_lock_hold = robot.data.joint_pos[0, _body_lock_joint_ids].clone()
    _right_arm_joint_ids = [robot.data.joint_names.index(n) for n in RIGHT_ARM_JOINTS]
    _right_hand_joint_ids = [robot.data.joint_names.index(n) for n in RIGHT_HAND_JOINTS]
    _right_arm_hold = robot.data.joint_pos[0, _right_arm_joint_ids].clone()
    _right_hand_hold = robot.data.joint_pos[0, _right_hand_joint_ids].clone()
    _idle_right_pos[:], _idle_right_quat[:] = _right_wrist_pose()
    right_pos[:] = _idle_right_pos
    right_quat[:] = _idle_right_quat


def _snapshot_right_arm() -> None:
    """Freeze right arm at current joint angles (called when movement keys are released)."""
    global _right_arm_hold
    if _right_arm_joint_ids is None:
        return
    _right_arm_hold = env.scene["robot"].data.joint_pos[0, _right_arm_joint_ids].clone()


def _apply_fingers_if_grasping() -> None:
    """Fingers are NOT driven by Pink — only this path writes hand joint targets."""
    if (
        gripper_closed
        or _finger_ctrl.thumb_prep_u > 0.0
        or _finger_ctrl.pinch_u > 0.0
        or _finger_ctrl.finger_u > 0.0
    ):
        _finger_ctrl.apply(env.scene["robot"])


def _patch_pink_skip_when_idle() -> None:
    """Pink IK: arm only while moving; fingers via DirectFingerController every substep."""
    pink = env.action_manager.get_term("pink_ik_cfg")
    arm_joint_ids = list(pink._isaaclab_controlled_joint_ids)

    def _apply_actions():
        if not _arm_moving:
            _freeze_robot()
            return
        ik_joint_positions = pink._compute_ik_solutions()
        pink._asset.set_joint_position_target(ik_joint_positions, arm_joint_ids)
        _enforce_body_lock()
        _apply_fingers_if_grasping()

    pink.apply_actions = _apply_actions


def _enforce_body_lock() -> None:
    """Hold left arm, left hand, waist, and legs at reset pose (every physics substep)."""
    robot = env.scene["robot"]
    if _body_lock_joint_ids is None or _body_lock_hold is None:
        return
    zeros = torch.zeros_like(_body_lock_hold).unsqueeze(0)
    robot.set_joint_position_target(_body_lock_hold.unsqueeze(0), joint_ids=_body_lock_joint_ids)
    robot.set_joint_velocity_target(zeros, joint_ids=_body_lock_joint_ids)


def _enforce_right_arm() -> None:
    if _right_arm_joint_ids is None or _right_arm_hold is None:
        return
    robot = env.scene["robot"]
    robot.set_joint_position_target(_right_arm_hold.unsqueeze(0), joint_ids=_right_arm_joint_ids)
    robot.set_joint_velocity_target(
        torch.zeros(1, len(_right_arm_joint_ids), device=robot.device),
        joint_ids=_right_arm_joint_ids,
    )


def _enforce_right_hand() -> None:
    if _right_hand_joint_ids is None or _right_hand_hold is None:
        return
    env.scene["robot"].set_joint_position_target(
        _right_hand_hold.unsqueeze(0),
        joint_ids=_right_hand_joint_ids,
    )


def _freeze_robot() -> None:
    """No movement keys: hold body + right arm; fingers from DirectFingerController only."""
    _enforce_body_lock()
    _enforce_right_arm()
    if gripper_closed or _finger_ctrl.thumb_prep_u > 0.0:
        _apply_fingers_if_grasping()
    elif not gripper_closed:
        _enforce_right_hand()


def _build_hand_vector() -> np.ndarray:
    """Dummy 6-D hand slice for Pink process_action (fingers are not applied by Pink)."""
    return right_hand_vector_from_named_positions(compute_drive_targets(0.0))


def _reset_scene() -> None:
    global gripper_closed, gripper_tight, _arm_moving, _grasp_state
    env.reset()
    teleop.reset()
    gripper_closed = False
    gripper_tight = False
    _arm_moving = False
    _grasp_state = "open"
    _finger_ctrl.reset(env.scene["robot"])
    _cache_arm_holds()
    robot = env.scene["robot"]
    for jname in ("R_index_proximal_joint", "R_ring_proximal_joint", "R_pinky_proximal_joint"):
        jid = robot.data.joint_names.index(jname)
        print(f"[INFO] {jname} id={jid} pos={robot.data.joint_pos[0, jid].item():.3f}")
    print("[INFO] Scene reset. K: side-grasp (move hand within ~9cm of cylinder first).")


def _hand_open() -> None:
    global gripper_closed, gripper_tight
    gripper_closed = False
    gripper_tight = False
    print("[INFO] Hand open (N)")


def _hand_close() -> None:
    global gripper_closed, gripper_tight
    gripper_closed = True
    gripper_tight = False
    z = contact_zone_metrics(env)
    print(
        f"[INFO] Grasp close (K) — zone={z['contact_zone_label']} "
        f"dist={z['contact_dist_m']:.3f}m height_err={z['contact_height_err_m']:+.3f}m"
    )
    if not z["contact_zone_ok"]:
        print(f"[zone] hint: {contact_zone_hint(str(z['contact_zone_label']))}")
    if z["contact_zone_label"] == "too_close":
        print("[INFO] Hand too close — finger close capped to reduce penetration;微退后再 K")


def _hand_tight() -> None:
    global gripper_closed, gripper_tight
    gripper_closed = True
    gripper_tight = True
    print("[INFO] Tight grasp (J) — stronger squeeze: index/middle + thumb pitch + ring/pinky")


def _update_grasp_status() -> None:
    global _grasp_state, _status_counter
    if is_lift_grasped(env):
        state = "lift_ok"
    elif is_grasped(env):
        state = "grasp_ok"
    elif _finger_ctrl.thumb_prep_u > 0.05 or _finger_ctrl.pinch_u > 0.05:
        state = "closing"
    else:
        state = "open"

    if state != _grasp_state:
        m = grasp_metrics(env)
        pf = m["per_finger"]
        if state == "grasp_ok":
            print(
                f"[GRASP OK] dist={m['hand_object_dist_m']:.3f}m "
                f"four={m['four_finger_closure']:.2f} thumb={m['thumb_closure']:.2f} "
                f"idx={pf['R_index_proximal_joint']:.2f} mid={pf['R_middle_proximal_joint']:.2f} "
                f"ring={pf['R_ring_proximal_joint']:.2f} pinky={pf['R_pinky_proximal_joint']:.2f}"
            )
        elif state == "lift_ok":
            print(f"[LIFT OK] object_z={m['object_z']:.3f}m")
        elif state == "open" and _grasp_state in ("grasp_ok", "lift_ok", "closing"):
            print("[GRASP LOST]")
        _grasp_state = state

    _status_counter += 1
    if _status_counter % 30 == 0 and (
        gripper_closed or _finger_ctrl.thumb_prep_u > 0.01 or _finger_ctrl.pinch_u > 0.01
    ):
        robot = env.scene["robot"]
        m = grasp_metrics(env)
        pf = m["per_finger"]
        tgt = compute_finger_targets_from_state(
            _finger_ctrl.thumb_prep_u,
            _finger_ctrl.pinch_u,
            _finger_ctrl.finger_u,
            tight=gripper_tight,
        )
        ring_t = tgt["R_ring_proximal_joint"]
        pinky_t = tgt["R_pinky_proximal_joint"]
        ring_it = tgt["R_ring_intermediate_joint"]
        pinky_it = tgt["R_pinky_intermediate_joint"]
        ring_a = float(robot.data.joint_pos[0, robot.data.joint_names.index("R_ring_proximal_joint")].item())
        pinky_a = float(robot.data.joint_pos[0, robot.data.joint_names.index("R_pinky_proximal_joint")].item())
        ring_ia = float(robot.data.joint_pos[0, robot.data.joint_names.index("R_ring_intermediate_joint")].item())
        pinky_ia = float(robot.data.joint_pos[0, robot.data.joint_names.index("R_pinky_intermediate_joint")].item())
        z = contact_zone_metrics(env)
        print(
            f"[hand] prep={_finger_ctrl.thumb_prep_u:.2f} contact_ok={_finger_ctrl.contact_allowed} "
            f"close_cap={_finger_ctrl.closure_cap:.2f} thumb_cap={_finger_ctrl.thumb_cap:.2f} "
            f"pinch={_finger_ctrl.pinch_u:.2f} finger={_finger_ctrl.finger_u:.2f} "
            f"phase={_finger_ctrl.grasp_phase()} "
            f"tight={gripper_tight} dist={m['hand_object_dist_m']:.3f} "
            f"idx/mid/ring/pinky="
            f"{pf['R_index_proximal_joint']:.2f}/"
            f"{pf['R_middle_proximal_joint']:.2f}/"
            f"{pf['R_ring_proximal_joint']:.2f}/"
            f"{pf['R_pinky_proximal_joint']:.2f} "
            f"thumb_perp={tgt['R_thumb_proximal_yaw_joint']:.2f} "
            f"thumb_pitch={tgt['R_thumb_proximal_pitch_joint']:.2f} "
            f"ring_p/tip={ring_t:.2f}/{ring_it:.2f} act={ring_a:.2f}/{ring_ia:.2f} "
            f"pinky_p/tip={pinky_t:.2f}/{pinky_it:.2f} act={pinky_a:.2f}/{pinky_ia:.2f}"
        )
        print(
            f"[zone] {z['contact_zone_label']} score={z['contact_zone_score']:.2f} "
            f"dist={z['contact_dist_m']:.3f}m (target {CONTACT_DIST_MIN_M:.2f}-{CONTACT_DIST_MAX_M:.2f}, "
            f"ideal ~{CONTACT_DIST_IDEAL_M:.2f}) height_err={z['contact_height_err_m']:+.3f}m"
        )
        if z["contact_zone_label"] != "ideal":
            print(f"[zone] hint: {contact_zone_hint(str(z['contact_zone_label']))}")

    elif _status_counter % 30 == 0:
        z = contact_zone_metrics(env)
        if z["contact_dist_m"] < 0.12:
            print(
                f"[zone] {z['contact_zone_label']} dist={z['contact_dist_m']:.3f}m "
                f"height_err={z['contact_height_err_m']:+.3f}m — {contact_zone_hint(str(z['contact_zone_label']))}"
            )


teleop.add_callback("R", _reset_scene)
teleop.add_callback("N", _hand_open)
teleop.add_callback("K", _hand_close)
teleop.add_callback("J", _hand_tight)

print("=" * 60)
print("  Red Block Teleop - RIGHT ARM ONLY")
print("    W/S A/D Q/E move   Z/X T/G C/V rotate (right wrist)")
print("    K: side grasp when [zone] ideal — if ring/pinky miss, press E to raise wrist first")
print("    J: tighter squeeze (thumb capped separately to avoid penetration)")
print("    Right arm moves ONLY while movement keys are held (stops on release)")
print("  Click Isaac Sim window before typing. Do not press L.")
print("=" * 60)

_reset_scene()
_patch_pink_skip_when_idle()

with torch.inference_mode():
    while simulation_app.is_running():
        was_moving = _arm_moving
        cmd = teleop.advance().cpu().numpy()
        dx, dy, dz, droll, dpitch, dyaw = cmd[:6].astype(np.float64)
        moving = np.linalg.norm(cmd[:6]) > 1e-8
        _arm_moving = moving

        if moving:
            if not was_moving:
                right_pos[:], right_quat[:] = _right_wrist_pose()
            right_pos += [dx, dy, dz]
            _clamp_wrist_target(right_pos)
            delta_r = R.from_euler("xyz", [droll, dpitch, dyaw])
            current_r = R.from_quat([right_quat[1], right_quat[2], right_quat[3], right_quat[0]])
            new_r = current_r * delta_r
            q = new_r.as_quat()
            right_quat[:] = [q[3], q[0], q[1], q[2]]
        elif was_moving:
            _snapshot_right_arm()

        dist = grasp_metrics(env)["hand_object_dist_m"]
        zone = contact_zone_metrics(env)
        _finger_ctrl.set_hand_distance(dist)
        _finger_ctrl.set_contact_height_err(float(zone["contact_height_err_m"]))

        if gripper_closed or gripper_tight:
            _finger_ctrl.set_closed(True, tight=gripper_tight)
        else:
            _finger_ctrl.set_closed(False)
        if not gripper_closed:
            gripper_tight = False

        if (
            gripper_closed
            and _finger_ctrl.thumb_prep_u >= 1.0
            and not _finger_ctrl.contact_allowed
        ):
            if not _grasp_far_warned:
                print(
                    f"[WARN] Hand too far for finger close (dist={dist:.3f}m, need <{_GRASP_CLOSE_MAX_DIST:.3f}m). "
                    "Move closer with WASD/QE, keep K held."
                )
                _grasp_far_warned = True
        elif not gripper_closed or _finger_ctrl.contact_allowed:
            _grasp_far_warned = False

        if gripper_closed or _finger_ctrl.thumb_prep_u > 0.0 or _finger_ctrl.pinch_u > 0.0 or _finger_ctrl.finger_u > 0.0:
            _apply_fingers_if_grasping()

        hand = _build_hand_vector()
        action = np.concatenate([right_pos, right_quat, hand]).astype(np.float32)
        env.step(torch.from_numpy(action).unsqueeze(0).to(env.device))

        _update_grasp_status()

env.close()
simulation_app.close()
