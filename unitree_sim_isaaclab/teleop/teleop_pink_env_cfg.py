"""Red-block scene + Pink IK (right arm only; left arm locked)."""
import copy
import tempfile

import isaaclab.controllers.utils as ControllerUtils
import isaaclab.sim as sim_utils
import pinocchio as pin
from pink.tasks import FrameTask

from grasp_rl.env_cfg import OBJECT_REST_Z, OBJECT_SIZE, make_env_cfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.controllers.pink_ik import PinkIKControllerCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import G1_INSPIRE_FTP_CFG
from teleop.teleop_hand import PINK_HAND_JOINT_NAMES, PINK_RIGHT_HAND_DRIVE_JOINT_NAMES
from teleop.teleop_fingers import INSPIRE_DRIVE_JOINTS, RIGHT_HAND_JOINTS

LEFT_HAND_JOINTS = [n for n in PINK_HAND_JOINT_NAMES if n.startswith("L_")]

# Teleop object: radius matched to Inspire hand grasp geometry.
TELEOP_OBJECT_RADIUS = 0.025
TELEOP_OBJECT_HEIGHT = 0.13
_TABLE_SURFACE_Z = OBJECT_REST_Z - OBJECT_SIZE * 0.5
TELEOP_OBJECT_REST_Z = _TABLE_SURFACE_Z + TELEOP_OBJECT_HEIGHT * 0.5

RIGHT_HAND_JOINTS_CFG = list(PINK_RIGHT_HAND_DRIVE_JOINT_NAMES)

RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
]
LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
]

WAIST_JOINTS = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]

# G1 29-DOF legs (locked during teleop so only the right arm moves).
LEG_LOCK_JOINTS_EXPR = [
    ".*_hip_.*",
    ".*_knee_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
]

BODY_LOCK_JOINTS = LEFT_ARM_JOINTS + LEFT_HAND_JOINTS + WAIST_JOINTS

_LOCK_ACTUATOR = dict(
    effort_limit=300.0,
    velocity_limit=100.0,
    stiffness=10000.0,
    damping=1000.0,
)

# Match G129 waist lock (very high damping stops torso wobble when right arm moves).
_WAIST_LOCK_ACTUATOR = dict(
    effort_limit=1000.0,
    velocity_limit=0.0,
    stiffness={
        "waist_yaw_joint": 10000.0,
        "waist_roll_joint": 10000.0,
        "waist_pitch_joint": 10000.0,
    },
    damping={
        "waist_yaw_joint": 10000.0,
        "waist_roll_joint": 10000.0,
        "waist_pitch_joint": 10000.0,
    },
)

_STABLE_OBJECT_RIGID = sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=False,
    retain_accelerations=False,
    linear_damping=8.0,
    angular_damping=8.0,
    max_linear_velocity=0.3,
    max_angular_velocity=0.5,
    max_depenetration_velocity=0.015,
)
_STABLE_OBJECT_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    friction_combine_mode="max",
    restitution_combine_mode="min",
    static_friction=5.0,
    dynamic_friction=4.0,
    restitution=0.0,
)


@configclass
class TeleopActionsCfg:
    """Pink IK: right wrist + 6 Inspire drive joints (13-D action). Left side / waist / legs locked."""

    pink_ik_cfg = PinkInverseKinematicsActionCfg(
        pink_controlled_joint_names=RIGHT_ARM_JOINTS,
        hand_joint_names=list(PINK_RIGHT_HAND_DRIVE_JOINT_NAMES),
        target_eef_link_names={"right_wrist": "right_wrist_yaw_link"},
        asset_name="robot",
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name="pelvis",
            num_hand_joints=len(PINK_RIGHT_HAND_DRIVE_JOINT_NAMES),
            show_ik_warnings=False,
            fail_on_joint_limit_violation=False,
            variable_input_tasks=[
                FrameTask(
                    "right_wrist_yaw_link",
                    position_cost=8.0,
                    orientation_cost=2.0,
                    lm_damping=12.0,
                    gain=0.5,
                ),
            ],
            fixed_input_tasks=[],
            xr_enabled=False,
        ),
        enable_gravity_compensation=False,
    )


def _right_wrist_frame_from_urdf(urdf_path: str) -> str:
    model = pin.buildModelFromUrdf(urdf_path)
    frame_names = [model.frames[i].name for i in range(1, model.nframes)]
    matches = [n for n in frame_names if "right_wrist_yaw" in n and n.endswith("_link")]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one right wrist link frame in {urdf_path}, found: {matches}")
    return matches[0]


def _patch_pink_ik_right_arm(controller_cfg, urdf_path: str) -> None:
    right_frame = _right_wrist_frame_from_urdf(urdf_path)
    for task in controller_cfg.variable_input_tasks:
        if isinstance(task, FrameTask):
            task.frame = right_frame


def _tune_robot_for_teleop(env_cfg) -> None:
    """Right arm = official pick-place PD; left arm frozen at default pose."""
    robot = env_cfg.scene.robot
    ftp_arms = G1_INSPIRE_FTP_CFG.actuators["arms"]

    robot.spawn.rigid_props = robot.spawn.rigid_props.replace(disable_gravity=True)
    robot.spawn.articulation_props = robot.spawn.articulation_props.replace(
        enabled_self_collisions=False,
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=4,
    )

    actuators = dict(robot.actuators)
    for key in ("arms", "waist", "legs", "feet", "hands"):
        actuators.pop(key, None)
    actuators["left_arm_lock"] = ImplicitActuatorCfg(
        joint_names_expr=LEFT_ARM_JOINTS,
        **_LOCK_ACTUATOR,
    )
    actuators["left_hand_lock"] = ImplicitActuatorCfg(
        joint_names_expr=LEFT_HAND_JOINTS,
        **_LOCK_ACTUATOR,
    )
    actuators["waist_lock"] = ImplicitActuatorCfg(
        joint_names_expr=WAIST_JOINTS,
        **_WAIST_LOCK_ACTUATOR,
    )
    actuators["legs_lock"] = ImplicitActuatorCfg(
        joint_names_expr=LEG_LOCK_JOINTS_EXPR,
        **_LOCK_ACTUATOR,
    )
    actuators["right_arm"] = ImplicitActuatorCfg(
        joint_names_expr=RIGHT_ARM_JOINTS,
        effort_limit=ftp_arms.effort_limit,
        velocity_limit=ftp_arms.velocity_limit,
        stiffness=ftp_arms.stiffness,
        damping=ftp_arms.damping,
        armature=copy.deepcopy(ftp_arms.armature),
    )
    hands = ImplicitActuatorCfg(
        joint_names_expr=list(RIGHT_HAND_JOINTS),
        effort_limit=320.0,
        velocity_limit=50,
        stiffness={
            "R_index_proximal_joint": 450.0,
            "R_index_intermediate_joint": 560.0,
            "R_middle_proximal_joint": 450.0,
            "R_middle_intermediate_joint": 560.0,
            "R_ring_proximal_joint": 720.0,
            "R_ring_intermediate_joint": 700.0,
            "R_pinky_proximal_joint": 720.0,
            "R_pinky_intermediate_joint": 700.0,
            "R_thumb_proximal_pitch_joint": 700.0,
            "R_thumb_proximal_yaw_joint": 700.0,
            "R_thumb_intermediate_joint": 500.0,
            "R_thumb_distal_joint": 450.0,
        },
        damping={
            "R_index_proximal_joint": 50.0,
            "R_index_intermediate_joint": 95.0,
            "R_middle_proximal_joint": 50.0,
            "R_middle_intermediate_joint": 95.0,
            "R_ring_proximal_joint": 115.0,
            "R_ring_intermediate_joint": 180.0,
            "R_pinky_proximal_joint": 115.0,
            "R_pinky_intermediate_joint": 180.0,
            "R_thumb_proximal_pitch_joint": 65.0,
            "R_thumb_proximal_yaw_joint": 65.0,
            "R_thumb_intermediate_joint": 55.0,
            "R_thumb_distal_joint": 50.0,
        },
    )
    actuators["hands"] = hands
    robot.actuators = actuators


def _tune_sim_for_teleop(env_cfg) -> None:
    env_cfg.decimation = 6
    env_cfg.sim.dt = 1.0 / 120.0
    env_cfg.sim.render_interval = 2
    env_cfg.sim.physx.bounce_threshold_velocity = 0.15
    env_cfg.sim.physx.enable_ccd = True
    env_cfg.sim.physx.num_position_iterations = 16
    env_cfg.sim.physx.num_velocity_iterations = 4
    env_cfg.sim.physx.contact_offset = 0.014
    env_cfg.sim.physx.rest_offset = 0.0025


def _tune_object_for_teleop(env_cfg) -> None:
    obj = env_cfg.scene.object
    spawn = sim_utils.CylinderCfg(
        radius=TELEOP_OBJECT_RADIUS,
        height=TELEOP_OBJECT_HEIGHT,
        axis="Z",
        rigid_props=_STABLE_OBJECT_RIGID,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.12),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.012,
            rest_offset=0.0025,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.0),
        physics_material=_STABLE_OBJECT_MATERIAL,
    )
    init_state = obj.init_state.replace(pos=[-4.45, -4.05, TELEOP_OBJECT_REST_Z])
    env_cfg.scene.object = obj.replace(spawn=spawn, init_state=init_state)


def make_teleop_pink_env_cfg(num_envs: int = 1, sim_device: str = "cuda:0"):
    env_cfg = make_env_cfg(num_envs, sim_device)
    _tune_robot_for_teleop(env_cfg)
    _tune_sim_for_teleop(env_cfg)
    _tune_object_for_teleop(env_cfg)
    env_cfg.actions = TeleopActionsCfg()
    env_cfg.terminations.time_out = None
    # Teleop: do not auto-reset on RL success/drop when lifting with Q after a grasp.
    env_cfg.terminations.task_success = None
    env_cfg.terminations.object_dropped = None

    temp_dir = tempfile.gettempdir()
    urdf_path, mesh_path = ControllerUtils.convert_usd_to_urdf(
        env_cfg.scene.robot.spawn.usd_path, temp_dir, force_conversion=True
    )
    pink = env_cfg.actions.pink_ik_cfg.controller
    pink.urdf_path = urdf_path
    pink.mesh_path = mesh_path
    _patch_pink_ik_right_arm(pink, urdf_path)
    return env_cfg
