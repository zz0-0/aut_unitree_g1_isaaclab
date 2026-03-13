import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

@configclass
class CameraBaseCfg:
    """camera base configuration class
    
    provide the default configuration for different types of cameras, support scene-specific parameter customization
    """
    
    @classmethod
    def get_camera_config(
        cls,
        prim_path: str = "/World/envs/env_.*/Robot/d435_link/front_cam",
        update_period: float = 0.02,
        height: int = 480,
        width: int =  640,
        focal_length: float = 7.6,
        focus_distance: float = 400.0,
        horizontal_aperture: float = 20.0,
        clipping_range: tuple = (0.1, 1.0e5),
        pos_offset: tuple = (0, 0.0, 0),
        rot_offset: tuple = (0.5, -0.5, 0.5, -0.5),
        data_types: list = []
    ) -> CameraCfg:
        """get the front camera configuration
        
        Args:
            prim_path: the path of the camera in the scene
            update_period: update period (seconds)
            height: image height (pixels)
            width: image width (pixels)
            focal_length: focal length
            focus_distance: focus distance
            horizontal_aperture: horizontal aperture
            clipping_range: clipping range (near clipping plane, far clipping plane)
            pos_offset: position offset (x, y, z)
            rot_offset: rotation offset quaternion
            data_types: data type list
            
        Returns:
            CameraCfg: camera configuration
        """
        if data_types is None:
            data_types = ["rgb"]

        return CameraCfg(
            prim_path=prim_path,
            update_period=update_period,
            height=height,
            width=width,
            data_types=data_types,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal_length,
                focus_distance=focus_distance,
                horizontal_aperture=horizontal_aperture,
                clipping_range=clipping_range
            ),
            offset=CameraCfg.OffsetCfg(
                pos=pos_offset,
                rot=rot_offset,
                convention="ros"
            )
        )
    
@configclass
class CameraPresets:
    @classmethod
    def g1_front_camera(cls) -> CameraCfg:
        """front camera configuration"""
        return CameraBaseCfg.get_camera_config(data_types=["rgb"])
    
    @classmethod
    def g1_world_camera(cls) -> CameraCfg:
        """world camera configuration"""
        return CameraBaseCfg.get_camera_config(
            prim_path="/World/envs/env_.*/Robot/d435_link/PerspectiveCamera_robot",
            pos_offset=(-0.9, 0.0, 0.0),
            rot_offset=(-0.51292, 0.51292, -0.48674, 0.48674),
            focal_length=12,
            horizontal_aperture=27,
            data_types=["rgb"]
        )