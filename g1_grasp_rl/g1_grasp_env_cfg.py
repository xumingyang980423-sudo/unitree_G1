"""G1 grasp-and-lift RL env config - joint position control (no PINK IK)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab_assets.robots.unitree import G1_INSPIRE_FTP_CFG

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ── custom MDP functions (inlined to avoid import issues) ──

def _object_is_lifted(env: ManagerBasedRLEnv, minimal_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    obj: RigidObject = env.scene[asset_cfg.name]
    return torch.where(obj.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def _eef_to_object_distance(env: ManagerBasedRLEnv, std: float, body_name: str = "right_wrist_yaw_link") -> torch.Tensor:
    obj: RigidObject = env.scene["object"]
    robot = env.scene["robot"]
    ee_idx = robot.data.body_names.index(body_name)
    ee_pos = robot.data.body_pos_w[:, ee_idx]
    dist = torch.norm(obj.data.root_pos_w - ee_pos, dim=1)
    return 1.0 - torch.tanh(dist / std)


def _eef_to_object_distance_raw(env: ManagerBasedRLEnv, body_name: str = "right_wrist_yaw_link") -> torch.Tensor:
    obj: RigidObject = env.scene["object"]
    robot = env.scene["robot"]
    ee_idx = robot.data.body_names.index(body_name)
    ee_pos = robot.data.body_pos_w[:, ee_idx]
    return obj.data.root_pos_w - ee_pos


# ── scene ──

@configclass
class GraspSceneCfg(InteractiveSceneCfg):
    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.30, 0.50, 1.04], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.10, 0.10, 0.10),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.15, 0.15)),
        ),
    )
    robot: ArticulationCfg = G1_INSPIRE_FTP_CFG.copy()
    robot.prim_path = "/World/envs/env_.*/Robot"
    robot.spawn.rigid_props.disable_gravity = False
    robot.spawn.articulation_props.fix_root_link = True
    robot.init_state = ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 1.0), rot=(0.7071, 0, 0, 0.7071),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    )
    robot.spawn.activate_contact_sensors = True
    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2000.0),
    )


# ── actions ──

@configclass
class ActionsCfg:
    joint_pos = base_mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.15, use_default_offset=True)


# ── observations ──

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=base_mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=base_mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        actions = ObsTerm(func=base_mdp.last_action)
        object_pos_w = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
        object_quat_w = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object")})
        right_eef_to_obj = ObsTerm(func=_eef_to_object_distance_raw)
        def __post_init__(self):
            self.enable_corruption = False; self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()


# ── rewards ──

@configclass
class RewardsCfg:
    reaching_object = RewTerm(func=_eef_to_object_distance, weight=3.0, params={"std": 0.10})
    lifting_object = RewTerm(func=_object_is_lifted, weight=20.0, params={"minimal_height": 0.15})
    action_rate_l2 = RewTerm(func=base_mdp.action_rate_l2, weight=-0.05)
    joint_vel_l2 = RewTerm(func=base_mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})


# ── terminations ──

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    object_dropping = DoneTerm(
        func=base_mdp.root_height_below_minimum,
        params={"minimum_height": 0.2, "asset_cfg": SceneEntityCfg("object")},
    )


# ── events ──

@configclass
class EventCfg:
    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")
    reset_object = EventTerm(
        func=base_mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {"x": [-0.05, 0.05], "y": [-0.05, 0.05]}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("object")},
    )


# ── main config ──

@configclass
class G1GraspRLEnvCfg(ManagerBasedRLEnvCfg):
    scene: GraspSceneCfg = GraspSceneCfg(num_envs=64, env_spacing=2.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    commands = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 12.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
