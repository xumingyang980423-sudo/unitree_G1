"""G1 Inspire Hand grasp-and-lift RL env (right arm + right hand)."""
from grasp_rl._paths import setup_repo_paths

_project_root = setup_repo_paths()  # repo root; used for assets/ paths below

import torch
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import configclass

from grasp_rl.coupled_grasp_action import (
    DualGroupGraspActionCfg,
    FOUR_FINGER_JOINT_NAMES,
    GRASP_ENABLE_DISTANCE,
    GRASP_GATE_STD,
    THUMB_JOINT_NAMES,
    _FOUR_FINGER_SPEC,
    _THUMB_SPEC,
)
from robots.unitree import G129_CFG_WITH_INSPIRE_HAND

# Per-step joint delta (rad): target = current_pos + scale * action.
ARM_ACTION_SCALE = 0.04
FINGER_ACTION_SCALE = 0.12  # slightly faster finger/thumb closing near the block

# Object geometry (4 cm cube — easier for Inspire hand to wrap than 6 cm).
OBJECT_SIZE = 0.04
OBJECT_HALF = OBJECT_SIZE * 0.5

# Pick-place: right arm + right Inspire hand only; left side stays at default pose.
MANIPULATION_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "R_index_proximal_joint",
    "R_index_intermediate_joint",
    "R_middle_proximal_joint",
    "R_middle_intermediate_joint",
    "R_pinky_proximal_joint",
    "R_pinky_intermediate_joint",
    "R_ring_proximal_joint",
    "R_ring_intermediate_joint",
    "R_thumb_proximal_yaw_joint",
    "R_thumb_proximal_pitch_joint",
    "R_thumb_intermediate_joint",
    "R_thumb_distal_joint",
]

# Object rest pose (center of cube on table).
OBJECT_REST_Z = 0.83
LIFT_HEIGHT = 0.05  # success when block center is 5 cm above rest height
LIFT_TARGET_Z = OBJECT_REST_Z + LIFT_HEIGHT  # 0.89 m
TABLE_DROP_Z = 0.75  # terminate if block falls off table

RIGHT_FINGER_JOINT_NAMES = FOUR_FINGER_JOINT_NAMES + THUMB_JOINT_NAMES
FOUR_FINGER_PROXIMAL_NAMES = [
    "R_index_proximal_joint",
    "R_middle_proximal_joint",
    "R_pinky_proximal_joint",
    "R_ring_proximal_joint",
]
THUMB_JOINT_NAMES_LIST = THUMB_JOINT_NAMES
ARM_JOINT_NAMES = MANIPULATION_JOINT_NAMES[:7]
EE_BODY_NAME = "right_wrist_yaw_link"
# Bodies used to estimate where the palm / fingertips are (not the wrist).
_FINGER_BODY_PREFIXES = ("R_index", "R_middle", "R_ring", "R_pinky")
_FINGER_BODY_SUFFIXES = ("distal", "intermediate", "proximal")

# Inspire hand: low joint angle = open, high = closed (see dds/inspire_dds.py denormalize).
_RIGHT_FINGER_CLOSE_RANGE = {
    "R_index_proximal_joint": (0.0, 1.7),
    "R_index_intermediate_joint": (0.0, 1.7),
    "R_middle_proximal_joint": (0.0, 1.7),
    "R_middle_intermediate_joint": (0.0, 1.7),
    "R_pinky_proximal_joint": (0.0, 0.5),
    "R_pinky_intermediate_joint": (0.0, 0.5),
    "R_ring_proximal_joint": (0.0, 0.5),
    "R_ring_intermediate_joint": (0.0, 0.5),
    "R_thumb_proximal_yaw_joint": (-0.1, 1.3),
    "R_thumb_proximal_pitch_joint": (-0.1, 1.3),
    "R_thumb_intermediate_joint": (-0.1, 1.3),
    "R_thumb_distal_joint": (-0.1, 1.3),
}


def _right_wrist_pos(env):
    robot = env.scene["robot"]
    idx = robot.data.body_names.index(EE_BODY_NAME)
    return robot.data.body_pos_w[:, idx]


def _right_hand_approach_pos(env):
    """Palm / fingertip centroid — avoids rewarding wrist-on-block collisions."""
    robot = env.scene["robot"]
    tips = []
    for name in robot.data.body_names:
        if not any(name.startswith(p) for p in _FINGER_BODY_PREFIXES):
            continue
        if not any(s in name for s in _FINGER_BODY_SUFFIXES):
            continue
        tips.append(robot.data.body_pos_w[:, robot.data.body_names.index(name)])
    if tips:
        return torch.stack(tips, dim=1).mean(dim=1)
    # Fallback: wrist offset toward the object (rough palm proxy).
    wrist = _right_wrist_pos(env)
    obj = env.scene["object"].data.root_pos_w
    direction = obj - wrist
    direction = direction / (torch.norm(direction, dim=1, keepdim=True) + 1e-6)
    return wrist + 0.06 * direction


def _object_to_hand_vec(env):
    return env.scene["object"].data.root_pos_w - _right_hand_approach_pos(env)


def _object_to_wrist_vec(env):
    return env.scene["object"].data.root_pos_w - _right_wrist_pos(env)


def _reach_object(env, std: float = 0.10):
    dist = torch.norm(_object_to_hand_vec(env), dim=1)
    return 1.0 - torch.tanh(dist / std)


def _approach_from_above(env, xy_std: float = 0.08, z_target: float = 0.05, z_std: float = 0.04):
    """Hand above block center, then descend — not wrist sideways into the cube."""
    hand = _right_hand_approach_pos(env)
    obj = env.scene["object"].data.root_pos_w
    xy = torch.norm(hand[:, :2] - obj[:, :2], dim=1)
    z_above = hand[:, 2] - obj[:, 2]
    xy_term = torch.exp(-xy / xy_std)
    z_term = torch.exp(-torch.abs(z_above - z_target) / z_std)
    return xy_term * z_term


def _hand_distance_penalty(env):
    return torch.norm(_object_to_hand_vec(env), dim=1)


def _wrist_distance_penalty(env):
    """Penalize wrist-object distance directly (strong gradient vs tanh reach alone)."""
    return torch.norm(_object_to_wrist_vec(env), dim=1)


def _wrist_overshoot_penalty(env, collision_dist: float = 0.05):
    """Penalize wrist touching block before fingers close."""
    wrist_dist = torch.norm(_object_to_wrist_vec(env), dim=1)
    closed = _four_finger_closure(env)
    return (wrist_dist < collision_dist).float() * (1.0 - closed)


def _four_finger_closure(env):
    """Normalized closure of the 4-finger group (proximal joints only)."""
    robot = env.scene["robot"]
    closures = []
    for name in FOUR_FINGER_PROXIMAL_NAMES:
        lo, hi = _RIGHT_FINGER_CLOSE_RANGE[name]
        pos = robot.data.joint_pos[:, robot.data.joint_names.index(name)]
        closures.append(torch.clamp((pos - lo) / (hi - lo + 1e-6), 0.0, 1.0))
    return torch.stack(closures, dim=1).mean(dim=1)


def _thumb_closure(env):
    robot = env.scene["robot"]
    closures = []
    for name in THUMB_JOINT_NAMES_LIST:
        lo, hi = _RIGHT_FINGER_CLOSE_RANGE[name]
        pos = robot.data.joint_pos[:, robot.data.joint_names.index(name)]
        closures.append(torch.clamp((pos - lo) / (hi - lo + 1e-6), 0.0, 1.0))
    return torch.stack(closures, dim=1).mean(dim=1)


def _right_finger_closure(env):
    """Normalized finger closure in [0, 1] using per-joint Inspire ranges."""
    robot = env.scene["robot"]
    indices = [robot.data.joint_names.index(n) for n in RIGHT_FINGER_JOINT_NAMES]
    idx = torch.tensor(indices, device=env.device).unsqueeze(0).expand(env.num_envs, -1)
    finger_pos = torch.gather(robot.data.joint_pos, 1, idx)

    closures = []
    for i, name in enumerate(RIGHT_FINGER_JOINT_NAMES):
        lo, hi = _RIGHT_FINGER_CLOSE_RANGE[name]
        closures.append(torch.clamp((finger_pos[:, i] - lo) / (hi - lo + 1e-6), 0.0, 1.0))
    return torch.stack(closures, dim=1).mean(dim=1)


def _near_object_soft(env, std: float = 0.10):
    """Soft proximity gate: 1 when hand is on block, decays with distance."""
    dist = torch.norm(_object_to_hand_vec(env), dim=1)
    return torch.exp(-dist / std)


def _grasp_finger_closure(env, near_std: float = 0.10):
    """Stage 2: four fingers + thumb close when hand is near the block."""
    near = _near_object_soft(env, near_std)
    return near * (0.65 * _four_finger_closure(env) + 0.35 * _thumb_closure(env))


def _opposition_grasp(env, near_std: float = 0.10):
    """Reward four fingers AND thumb closing together (pinch / power grasp)."""
    near = _near_object_soft(env, near_std)
    four = _four_finger_closure(env)
    thumb = _thumb_closure(env)
    return near * four * thumb


def _penalize_four_without_thumb(env, near_std: float = 0.10, min_four: float = 0.30):
    """Discourage curling four fingers while thumb stays open near the block."""
    near = _near_object_soft(env, near_std)
    four = _four_finger_closure(env)
    thumb = _thumb_closure(env)
    gap = (four - thumb).clamp(min=0.0)
    active = (four > min_four).float()
    return near * active * gap


def _penalize_close_when_far(env, far_dist: float = 0.14):
    """Penalize finger/thumb closure while hand is still far from the block."""
    dist = torch.norm(_object_to_hand_vec(env), dim=1)
    far = (dist > far_dist).float()
    closed = _right_finger_closure(env)
    return far * closed


def _pregrasp_open_when_far(env, far_std: float = 0.12):
    """Stage 1: keep hand open while approaching."""
    far = 1.0 - _near_object_soft(env, far_std)
    four_open = (1.0 - _four_finger_closure(env)).clamp(0.0, 1.0)
    thumb_open = (1.0 - _thumb_closure(env)).clamp(0.0, 1.0)
    return far * (0.6 * four_open + 0.4 * thumb_open)


def _finger_motion_near(env, near_std: float = 0.12, vel_scale: float = 3.0):
    """Stage 2: reward finger joint motion while the wrist is near the block."""
    near = _near_object_soft(env, near_std)
    robot = env.scene["robot"]
    indices = [robot.data.joint_names.index(n) for n in RIGHT_FINGER_JOINT_NAMES]
    idx = torch.tensor(indices, device=env.device).unsqueeze(0).expand(env.num_envs, -1)
    finger_vel = torch.gather(robot.data.joint_vel, 1, idx)
    motion = torch.mean(torch.abs(finger_vel), dim=1)
    return near * torch.tanh(motion / vel_scale)


def _grasp_firm(env, near_dist: float = 0.10, min_four: float = 0.40, min_thumb: float = 0.30):
    """Stage 3: firm grasp — four fingers and thumb both closed near the block."""
    dist = torch.norm(_object_to_hand_vec(env), dim=1)
    near = (dist < near_dist).float()
    four = (_four_finger_closure(env) >= min_four).float()
    thumb = (_thumb_closure(env) >= min_thumb).float()
    return near * four * thumb


def _carry_while_lifted(env, near_dist: float = 0.12, min_closure: float = 0.40, min_z_delta: float = 0.015):
    """Stage 4: block stays with the hand while lifted (not left on the table)."""
    z = env.scene["object"].data.root_pos_w[:, 2]
    lifted = (z > OBJECT_REST_Z + min_z_delta).float()
    dist = torch.norm(_object_to_hand_vec(env), dim=1)
    near = (dist < near_dist).float()
    closed = (_right_finger_closure(env) >= min_closure).float()
    return lifted * near * closed


def _lift_height_shaping(env, near_dist: float = 0.10, min_closure: float = 0.25):
    """Stage 4 (dense): lift progress only after a firm grasp."""
    z = env.scene["object"].data.root_pos_w[:, 2]
    progress = torch.clamp((z - OBJECT_REST_Z) / LIFT_HEIGHT, 0.0, 1.0)

    dist = torch.norm(_object_to_hand_vec(env), dim=1)
    near = (dist < near_dist).float()
    closed = (_right_finger_closure(env) >= min_closure).float()
    return progress * near * closed


def _lift_success(env, near_dist: float = 0.10, min_closure: float = 0.50):
    """Sparse lift success only when grasped (near hand + fingers closed)."""
    z = env.scene["object"].data.root_pos_w[:, 2]
    lifted = (z >= LIFT_TARGET_Z).float()

    dist = torch.norm(_object_to_hand_vec(env), dim=1)
    near = (dist < near_dist).float()
    closed = (_right_finger_closure(env) >= min_closure).float()
    return lifted * near * closed


def _grasp_and_lift(env, near_dist: float = 0.10, min_closure: float = 0.50):
    """Bonus when the block is grasped AND lifted."""
    dist = torch.norm(_object_to_hand_vec(env), dim=1)
    lifted = env.scene["object"].data.root_pos_w[:, 2] >= LIFT_TARGET_Z
    closed = _right_finger_closure(env) >= min_closure
    return ((dist < near_dist) & closed & lifted).float()


def _penalize_open_hand_lift(env, near_dist: float = 0.08, min_z_delta: float = 0.025):
    """Discourage lifting the block without closing fingers (only when clearly lifted)."""
    z = env.scene["object"].data.root_pos_w[:, 2]
    lifted_clearly = (z > OBJECT_REST_Z + min_z_delta).float()

    dist = torch.norm(_object_to_hand_vec(env), dim=1)
    near = (dist < near_dist).float()
    open_hand = (1.0 - _right_finger_closure(env)).clamp(0.0, 1.0)
    return lifted_clearly * near * open_hand


def _finger_closure_uniformity(env, near_std: float = 0.12):
    """Reward identical closure across the 4 proximal fingers when near the block."""
    near = _near_object_soft(env, near_std)
    closures = []
    robot = env.scene["robot"]
    for name in FOUR_FINGER_PROXIMAL_NAMES:
        lo, hi = _RIGHT_FINGER_CLOSE_RANGE[name]
        pos = robot.data.joint_pos[:, robot.data.joint_names.index(name)]
        closures.append(torch.clamp((pos - lo) / (hi - lo + 1e-6), 0.0, 1.0))
    closure_stack = torch.stack(closures, dim=1)
    std = closure_stack.std(dim=1)
    return near * torch.exp(-8.0 * std)


def _object_dropped(env):
    return env.scene["object"].data.root_pos_w[:, 2] < TABLE_DROP_Z


def _task_success(env, near_dist: float = 0.12, min_closure: float = 0.45):
    """Episode success: block lifted 5 cm while grasped."""
    z = env.scene["object"].data.root_pos_w[:, 2]
    lifted = z >= LIFT_TARGET_Z
    dist = torch.norm(_object_to_hand_vec(env), dim=1)
    near = dist < near_dist
    closed = _right_finger_closure(env) >= min_closure
    return lifted & near & closed


_robot_cfg = G129_CFG_WITH_INSPIRE_HAND.replace(
    prim_path="/World/envs/env_.*/Robot",
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-4.2, -3.7, 0.76),
        rot=(0.7071, 0, 0, -0.7071),
        joint_pos={
            **G129_CFG_WITH_INSPIRE_HAND.init_state.joint_pos,
            # Start more open so the policy learns a clear close motion.
            "R_index_proximal_joint": 0.05,
            "R_index_intermediate_joint": 0.04,
            "R_middle_proximal_joint": 0.05,
            "R_middle_intermediate_joint": 0.04,
            "R_pinky_proximal_joint": 0.02,
            "R_pinky_intermediate_joint": 0.02,
            "R_ring_proximal_joint": 0.02,
            "R_ring_intermediate_joint": 0.02,
            "R_thumb_proximal_yaw_joint": 0.08,
            "R_thumb_proximal_pitch_joint": 0.05,
            "R_thumb_intermediate_joint": 0.04,
            "R_thumb_distal_joint": 0.03,
        },
        joint_vel={".*": 0.0},
    ),
)
_robot_cfg.spawn.articulation_props.fix_root_link = True

# Softer finger PD gains reduce high-frequency finger vibration during RL.
_hands_cfg = ImplicitActuatorCfg(
    joint_names_expr=[
        ".*_index_proximal_joint", ".*_index_intermediate_joint",
        ".*_middle_proximal_joint", ".*_middle_intermediate_joint",
        ".*_pinky_proximal_joint", ".*_pinky_intermediate_joint",
        ".*_ring_proximal_joint", ".*_ring_intermediate_joint",
        ".*_thumb_proximal_yaw_joint", ".*_thumb_proximal_pitch_joint",
        ".*_thumb_intermediate_joint", ".*_thumb_distal_joint",
    ],
    effort_limit=100.0,
    velocity_limit=50,
    stiffness={".*": 150.0},
    damping={".*": 25.0},
)
_robot_cfg = _robot_cfg.replace(actuators={**_robot_cfg.actuators, "hands": _hands_cfg})


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Minimal grasp-lift scene: ground + table + robot + red block."""

    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=4000.0),
    )
    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-4.3, -4.2, -0.2], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(usd_path=f"{_project_root}/assets/objects/table_with_yellowbox.usd"),
    )
    object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-4.25, -4.03, OBJECT_REST_Z], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(OBJECT_SIZE, OBJECT_SIZE, OBJECT_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, retain_accelerations=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, contact_offset=0.01, rest_offset=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=10,
                dynamic_friction=1.5,
                restitution=0.01,
            ),
        ),
    )
    robot: ArticulationCfg = _robot_cfg


@configclass
class ActionsCfg:
    # 7-D arm + 2-D grasp (four fingers as one unit + opposing thumb).
    arm = base_mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        scale=ARM_ACTION_SCALE,
        use_zero_offset=True,
        clip={".*": (-1.0, 1.0)},
    )
    grasp = DualGroupGraspActionCfg(
        asset_name="robot",
        finger_joint_names=FOUR_FINGER_JOINT_NAMES,
        thumb_joint_names=THUMB_JOINT_NAMES,
        finger_spec=_FOUR_FINGER_SPEC,
        thumb_spec=_THUMB_SPEC,
        scale=FINGER_ACTION_SCALE,
        grasp_enable_distance=GRASP_ENABLE_DISTANCE,
        grasp_gate_std=GRASP_GATE_STD,
    )


@configclass
class ObsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_pos = ObservationTermCfg(
            func=base_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=MANIPULATION_JOINT_NAMES)},
        )
        joint_vel = ObservationTermCfg(
            func=base_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=MANIPULATION_JOINT_NAMES)},
        )
        actions = ObservationTermCfg(func=base_mdp.last_action)
        object_pos = ObservationTermCfg(
            func=base_mdp.root_pos_w,
            params={"asset_cfg": SceneEntityCfg("object")},
        )
        object_to_hand = ObservationTermCfg(func=_object_to_hand_vec)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    # --- Stage 1: approach with open hand ---
    pregrasp_open = RewardTermCfg(func=_pregrasp_open_when_far, weight=4.0, params={"far_std": 0.12})
    close_when_far = RewardTermCfg(func=_penalize_close_when_far, weight=-5.0, params={"far_dist": 0.14})
    reach = RewardTermCfg(func=_reach_object, weight=8.0, params={"std": 0.10})
    approach_above = RewardTermCfg(
        func=_approach_from_above, weight=10.0, params={"xy_std": 0.07, "z_target": 0.04, "z_std": 0.035}
    )
    hand_distance = RewardTermCfg(func=_hand_distance_penalty, weight=-5.0)
    wrist_overshoot = RewardTermCfg(
        func=_wrist_overshoot_penalty, weight=-10.0, params={"collision_dist": 0.045}
    )
    # --- Stage 2: four fingers + thumb close together when hand is near ---
    finger_motion = RewardTermCfg(
        func=_finger_motion_near, weight=5.0, params={"near_std": 0.10, "vel_scale": 2.5}
    )
    grasp = RewardTermCfg(func=_grasp_finger_closure, weight=14.0, params={"near_std": 0.10})
    opposition = RewardTermCfg(func=_opposition_grasp, weight=18.0, params={"near_std": 0.10})
    four_without_thumb = RewardTermCfg(
        func=_penalize_four_without_thumb, weight=-6.0, params={"near_std": 0.10, "min_four": 0.30}
    )
    # --- Stage 3: firm grasp (four fingers + thumb wrap block) ---
    grasp_firm = RewardTermCfg(
        func=_grasp_firm, weight=18.0, params={"near_dist": 0.09, "min_four": 0.35, "min_thumb": 0.28}
    )
    finger_uniformity = RewardTermCfg(
        func=_finger_closure_uniformity, weight=8.0, params={"near_std": 0.12}
    )
    # --- Stage 4: arm lifts and block moves up with the hand ---
    lift_shaping = RewardTermCfg(
        func=_lift_height_shaping, weight=20.0, params={"near_dist": 0.10, "min_closure": 0.40}
    )
    carry = RewardTermCfg(
        func=_carry_while_lifted, weight=10.0, params={"near_dist": 0.12, "min_closure": 0.40, "min_z_delta": 0.015}
    )
    lift_success = RewardTermCfg(
        func=_lift_success, weight=40.0, params={"near_dist": 0.10, "min_closure": 0.50}
    )
    grasp_lift = RewardTermCfg(
        func=_grasp_and_lift, weight=20.0, params={"near_dist": 0.10, "min_closure": 0.50}
    )
    open_hand_lift = RewardTermCfg(
        func=_penalize_open_hand_lift, weight=-4.0, params={"near_dist": 0.10, "min_z_delta": 0.02}
    )
    # Smooth motion penalties (slightly reduced so finger closing is not over-penalized).
    action_rate = RewardTermCfg(func=base_mdp.action_rate_l2, weight=-0.3)
    joint_vel = RewardTermCfg(
        func=base_mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=MANIPULATION_JOINT_NAMES)},
    )


@configclass
class TermCfg:
    time_out = TerminationTermCfg(func=base_mdp.time_out, time_out=True)
    object_dropped = TerminationTermCfg(func=_object_dropped)
    task_success = TerminationTermCfg(func=_task_success, time_out=False)


@configclass
class EvtCfg:
    reset_scene = EventTermCfg(func=base_mdp.reset_scene_to_default, mode="reset")
    reset_object = EventTermCfg(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [-0.05, 0.05], "y": [-0.05, 0.05]},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


def make_env_cfg(num_envs: int = 1, sim_device: str = "cuda:0"):
    @configclass
    class EnvCfg(ManagerBasedRLEnvCfg):
        scene: SceneCfg = SceneCfg(num_envs=num_envs, env_spacing=2.5, replicate_physics=True)
        observations: ObsCfg = ObsCfg()
        actions: ActionsCfg = ActionsCfg()
        terminations = TermCfg()
        rewards = RewardsCfg()
        events = EvtCfg()
        commands = None
        curriculum = None
        viewer: ViewerCfg = ViewerCfg(eye=(-2.0, -2.0, 2.0), lookat=(-4.2, -4.0, 0.8))

        def __post_init__(self):
            # 25 Hz control for smoother, more human-like arm motion.
            self.decimation = 8
            self.episode_length_s = 15.0
            self.sim.dt = 0.005
            self.sim.render_interval = self.decimation
            self.sim.device = sim_device
            self.sim.physx.bounce_threshold_velocity = 0.01
            self.sim.physx.enable_ccd = True
            self.sim.physx.num_substeps = 2
            self.sim.physx.num_position_iterations = 12
            self.sim.physx.num_velocity_iterations = 4
            self.sim.physx.contact_offset = 0.015
            self.sim.physx.rest_offset = 0.001
            self.sim.physx.gpu_constraint_solver_heavy_spring_enabled = True

    return EnvCfg()
