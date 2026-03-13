from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
import isaaclab.sim as sim_utils
from aut_unitree_g1_isaaclab.assets.robot.unitree import UNITREE_G1_29DOF_CFG_WITH_INSPIRE_HAND, UNITREE_G1_29DOF_CFG_WITH_INSPIRE_HAND_WHOLEBODY, UNITREE_G1_29DOF_CFG_RLLAB
from typing import Optional, Dict, Tuple, Literal

@configclass
class RobotJointTemplates:
    """G1 robot joint template collection
    
    provide different types of joint configuration templates
    """
    
    @classmethod
    def get_leg_joints(cls) -> Dict[str, float]:
        """get the default position of the leg joints"""
        return {
            # left leg joint - locked in standing position
            "left_hip_pitch_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.0,
            "left_ankle_pitch_joint": 0.0,
            "left_ankle_roll_joint": 0.0,
            
            # right leg joint - locked in standing position
            "right_hip_pitch_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.0,
            "right_ankle_pitch_joint": 0.0,
            "right_ankle_roll_joint": 0.0,
        }
    
    @classmethod
    def get_waist_joints(cls, include_waist: bool = True) -> Dict[str, float]:
        """get the position of the waist joints
        
        Args:
            include_waist: whether to include the waist joint
            
        Returns:
            waist joint position dictionary, if not included, return an empty dictionary
        """
        if not include_waist:
            return {}
        
        return {
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
        }
    
    @classmethod
    def get_arm_joints(cls) -> Dict[str, float]:
        """get the default position of the arm joints
        
        ✅ CRITICAL FIX: Start with arms in natural "ready to reach" pose
        Instead of straight down (0.0), arms should be:
        - Slightly raised (shoulder pitch ~0.3-0.5)
        - Elbows bent (elbow ~1.5)
        - Hands forward of body
        
        This gives robot a GOOD STARTING POINT for reaching motions!
        """
        return {
            # Left arm - natural reaching pose
            "left_shoulder_pitch_joint": 0.3,   # Arm slightly forward
            "left_shoulder_roll_joint": 0.3,    # Arm slightly out to side
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 1.5,            # Elbow bent 90 degrees
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            
            # Right arm - natural reaching pose
            "right_shoulder_pitch_joint": 0.3,  # Arm slightly forward
            "right_shoulder_roll_joint": -0.3,  # Arm slightly out to side (negative for right)
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 1.5,           # Elbow bent 90 degrees
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        }
    
    @classmethod
    def get_hand_joints(cls, hand_type: Literal["gripper", "dex3","inspire"] = "gripper") -> Dict[str, float]:
        """get the default position of the hand joints
        
        Args:
            hand_type: hand type
                - "gripper": simple gripper (2 joints)
                - "dex3": dexterous hand (14 joints)
                
        Returns:
            hand joint position dictionary
        """
        if hand_type == "gripper":
            return {
                # simple gripper joint
                "left_hand_Joint1_1": 0.0,
                "left_hand_Joint2_1": 0.0,
                "right_hand_Joint1_1": 0.0,
                "right_hand_Joint2_1": 0.0,
            }
        elif hand_type == "dex3":
            return {
                # dexterous hand joint - left hand
                "left_hand_index_0_joint": 0.0,
                "left_hand_middle_0_joint": 0.0,
                "left_hand_thumb_0_joint": 0.0,
                "left_hand_index_1_joint": 0.0,
                "left_hand_middle_1_joint": 0.0,
                "left_hand_thumb_1_joint": 0.0,
                "left_hand_thumb_2_joint": 0.0,
                
                # dexterous hand joint - right hand
                "right_hand_index_0_joint": 0.0,
                "right_hand_middle_0_joint": 0.0,
                "right_hand_thumb_0_joint": 0.0,
                "right_hand_index_1_joint": 0.0,
                "right_hand_middle_1_joint": 0.0,
                "right_hand_thumb_1_joint": 0.0,
                "right_hand_thumb_2_joint": 0.0,
            }
        elif hand_type == "inspire":
            return {
            # fingers joints
            "L_index_proximal_joint": 0.0,
            "L_index_intermediate_joint": 0.0,
            "L_middle_proximal_joint": 0.0,
            "L_middle_intermediate_joint": 0.0,
            "L_pinky_proximal_joint":0.0,
            "L_pinky_intermediate_joint":0.0,
            "L_ring_proximal_joint":0.0,
            "L_ring_intermediate_joint":0.0,
            "L_thumb_proximal_yaw_joint":0.0,
            "L_thumb_proximal_pitch_joint":0.0,
            "L_thumb_intermediate_joint":0.0,
            "L_thumb_distal_joint":0.0,

            "R_index_proximal_joint": 0.0,
            "R_index_intermediate_joint": 0.0,
            "R_middle_proximal_joint": 0.0,
            "R_middle_intermediate_joint": 0.0,
            "R_pinky_proximal_joint":0.0,
            "R_pinky_intermediate_joint":0.0,
            "R_ring_proximal_joint":0.0,
            "R_ring_intermediate_joint":0.0,
            "R_thumb_proximal_yaw_joint":0.0,
            "R_thumb_proximal_pitch_joint":0.0,
            "R_thumb_intermediate_joint":0.0,
            "R_thumb_distal_joint":0.0,
            }
        else:
            raise ValueError(f"Unsupported hand type: {hand_type}. Supported: 'gripper', 'dex3'")


@configclass
class RobotBaseCfg:
    """G1 robot base configuration class
    
    provide the flexible configuration for G1 robot, support:
    - with/without waist joint
    - simple gripper/dexterous hand
    - basic articulation configuration
    - default joint position
    - scene-specific parameter customization
    """
    
    @classmethod
    def get_base_config(
        cls,
        prim_path: str = "/World/envs/env_.*/Robot",
        init_pos: Tuple[float, float, float] = (-0.15, 0.0, 0.744),
        init_rot: Tuple[float, float, float, float] = (0.7071, 0, 0, 0.7071),
        include_waist: bool = True,
        hand_type: Literal["gripper", "dex3", "inspire"] = "gripper",
        base_config = UNITREE_G1_29DOF_CFG_WITH_INSPIRE_HAND,
        custom_joint_pos: Optional[Dict[str, float]] = None,
        is_have_hand: bool = True,
        update_default_joint_pos: bool = True,
        robot_type: Literal["g129dof", "h1_2"] = "g129dof",
    ) -> ArticulationCfg:
        """get the base configuration for G1 robot
        
        Args:
            prim_path: the path of the robot in the scene
            init_pos: initial position (x, y, z)
            init_rot: initial rotation quaternion (w, x, y, z)
            include_waist: whether to include the waist joint
            hand_type: hand type ("simple" or "dexterous")
            base_config: base robot configuration, default using G129_CFG_WITH_DEX1_WAIST_FIX
            custom_joint_pos: custom joint position dictionary, will override the default value
            
        Returns:
            ArticulationCfg: robot configuration
        """
        
        if update_default_joint_pos:
            # build the complete default joint position
            default_joint_pos = {}
            # add the leg joints
            default_joint_pos.update(RobotJointTemplates.get_leg_joints())
            
            # add the waist joints (if enabled)
            if robot_type == "g129dof":
                default_joint_pos.update(RobotJointTemplates.get_waist_joints(include_waist))
            
            # add the arm joints
            default_joint_pos.update(RobotJointTemplates.get_arm_joints())
            
            # add the hand joints
            if is_have_hand:
                default_joint_pos.update(RobotJointTemplates.get_hand_joints(hand_type))
        else:
            default_joint_pos = base_config.init_state.joint_pos.copy()
        
        # if the custom joint position is provided, merge it
        if custom_joint_pos:
            joint_pos = {**default_joint_pos, **custom_joint_pos}
        else:
            joint_pos = default_joint_pos
        
        # create the base configuration
        return base_config.replace(
            prim_path=prim_path,
            init_state=ArticulationCfg.InitialStateCfg(
                pos=init_pos,
                rot=init_rot,
                joint_pos=joint_pos,
                joint_vel={".*": 0.0}
            ),
        )

@configclass
class G1RobotPresets:

    @classmethod
    def g1_29dof_inspire_base_fix(cls, init_pos, init_rot) -> ArticulationCfg:
        """pick-place task configuration - inspire hand"""
        return RobotBaseCfg.get_base_config(
            init_pos=init_pos,
            init_rot=init_rot,
            include_waist=True,
            hand_type="inspire",
            base_config=UNITREE_G1_29DOF_CFG_WITH_INSPIRE_HAND)
    
    @classmethod
    def g1_29dof_inspire_wholebody_cfg(cls, init_pos=None, init_rot=None) -> ArticulationCfg:
        """wholebody control task configuration - inspire hand
        
        Uses the original crouched pose from UNITREE_G1_29DOF_CFG_WITH_INSPIRE_HAND_WHOLEBODY
        which is optimized for locomotion (bent knees, proper CoM).
        """
        # Switch to RL-Lab floating-base config (no Inspire hand) to ensure base is free
        base_cfg = UNITREE_G1_29DOF_CFG_RLLAB
        if init_pos is None:
            init_pos = base_cfg.init_state.pos
        if init_rot is None:
            init_rot = base_cfg.init_state.rot if base_cfg.init_state.rot else (1.0, 0.0, 0.0, 0.0)
        
        return RobotBaseCfg.get_base_config(
            init_pos=init_pos,
            init_rot=init_rot,
            include_waist=True,
            hand_type="gripper",
            is_have_hand=False,
            base_config=base_cfg,
            update_default_joint_pos=False,  # Preserve RL-Lab crouched locomotion pose
        )
    
@configclass
class UnitreeArticulationCfg(ArticulationCfg):
    """Configuration for Unitree articulations."""

    joint_sdk_names: list[str] = None

    soft_joint_pos_limit_factor = 0.9

@configclass
class UnitreeUsdFileCfg(sim_utils.UsdFileCfg):
    activate_contact_sensors: bool = True
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
    )
