import isaaclab.sim as sim_utils
from isaaclab.assets import  AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from aut_unitree_g1_isaaclab.assets.flag import FLAG_CONFIG
from aut_unitree_g1_isaaclab.tasks.common_config import CameraBaseCfg

@configclass
class FlagSceneCfg(InteractiveSceneCfg):
    flag = FLAG_CONFIG

    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0))