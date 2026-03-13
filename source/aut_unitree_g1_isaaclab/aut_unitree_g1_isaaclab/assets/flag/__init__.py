from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
import isaaclab.sim as sim_utils

FLAG_CONFIG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Flag",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/ryz5920/aut_unitree_g1_isaaclab/source/aut_unitree_g1_isaaclab/aut_unitree_g1_isaaclab/assets/flag/flag.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=False,
                kinematic_enabled=True  # ✅ Flag is frozen in space, won't fall!
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.0
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0), metallic=0
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.4, -0.3, 0.85),  # 40cm in front, waist height
        rot=(1.0, 0.0, 0.0, 0.0),  # Upright orientation
    ),
)