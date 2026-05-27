"""View G1 at table with cube object - simplified, no physics."""
import argparse, os, sys

_ISAAC_SIM_PATH = "E:\\Issac_sim\\isaac-sim-standalone-5.1.0-windows-x86_64"
os.add_dll_directory(os.path.join(_ISAAC_SIM_PATH, "kit", "python", "Lib", "site-packages", "h5py"))
import h5py; del h5py
os.environ.setdefault("CARB_APP_PATH", os.path.join(_ISAAC_SIM_PATH, "kit"))
os.environ.setdefault("ISAAC_PATH", _ISAAC_SIM_PATH)
os.environ.setdefault("EXP_PATH", os.path.join(_ISAAC_SIM_PATH, "apps"))

import isaacsim
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.num_envs = 1

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

print("[INFO] Building scene: G1 + table + cube...")

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files import spawn_ground_plane, spawn_from_usd
from isaaclab.sim.spawners.lights import spawn_light
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.sim.spawners.shapes import spawn_cuboid, spawn_sphere

spawn_ground_plane("/World/ground", GroundPlaneCfg())
spawn_light("/World/light", sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0))

# Table
print("[INFO] Spawning table...")
spawn_from_usd(
    "/World/Table",
    UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    ),
    translation=(0.0, 0.55, 0.0),
    orientation=(1.0, 0.0, 0.0, 0.0),
)

# Red cube on table using Isaac Lab spawner
print("[INFO] Spawning red cube on table...")
spawn_cuboid(
    "/World/Cube",
    sim_utils.CuboidCfg(
        size=(0.12, 0.12, 0.12),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    ),
    translation=(-0.30, 0.45, 1.10),
)

# Green sphere on table
print("[INFO] Spawning green sphere on table...")
spawn_sphere(
    "/World/Sphere",
    sim_utils.SphereCfg(
        radius=0.06,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.3)),
    ),
    translation=(-0.15, 0.60, 1.10),
)

# G1 robot
print("[INFO] Spawning G1 robot...")
spawn_from_usd(
    "/World/Robot",
    UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=False),
    ),
    translation=(0.0, 0.0, 1.0),
    orientation=(0.7071, 0.0, 0.0, 0.7071),
)

print("=" * 60)
print("  G1 + table + red cube (above table)")
print("  Right-drag to orbit, scroll to zoom")
print("  Close window to exit")
print("=" * 60)

while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()
