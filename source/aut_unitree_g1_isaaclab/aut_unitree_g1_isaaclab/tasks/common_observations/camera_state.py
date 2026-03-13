# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""
camera state
"""     

from __future__ import annotations

from typing import TYPE_CHECKING
import torch
import sys
import os
import threading
import queue
import numpy as np

# add the project root directory to the path, so that the shared memory tool can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from image_server.shared_memory_utils import MultiImageWriter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# create the global multi-image shared memory writer
multi_image_writer = MultiImageWriter()

def set_writer_options(enable_jpeg: bool = False, jpeg_quality: int = 85, skip_cvtcolor: bool = False):
    try:
        multi_image_writer.set_options(enable_jpeg=enable_jpeg, jpeg_quality=jpeg_quality, skip_cvtcolor=skip_cvtcolor)
        print(f"[camera_state] writer options: jpeg={enable_jpeg}, quality={jpeg_quality}, skip_cvtcolor={skip_cvtcolor}")
    except Exception as e:
        print(f"[camera_state] failed to set writer options: {e}")


_camera_cache = {
    'available_cameras': None,
    'camera_keys': None,
    'last_scene_id': None,
    'frame_step': 0,
    'write_interval_steps': 2,
}


_return_placeholder = None
_async_queue = None
_async_thread = None
_async_started = False

def _async_writer_loop(q: "queue.Queue", writer: MultiImageWriter):
    while True:
        try:
            item = q.get()
            if item is None:
                break
            writer.write_images(item)
        except Exception as e:
            print(f"[camera_state] Async writer error: {e}")

def _ensure_async_started():
    global _async_started, _async_queue, _async_thread
    if not _async_started:
        _async_queue = queue.Queue(maxsize=1)
        _async_thread = threading.Thread(target=_async_writer_loop, args=(_async_queue, multi_image_writer), daemon=True)
        _async_thread.start()
        _async_started = True


def get_camera_image(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """get camera image flattened for RL observation space
    
    Args:
        env: ManagerBasedRLEnv - reinforcement learning environment instance
    
    Returns:
        torch.Tensor: Flattened camera image tensor with shape (batch, flattened_size)
    """
    global _return_placeholder
    
    # Get batch size from environment
    batch_size = env.num_envs
    
    # Initialize placeholder on the same device as the environment
    if _return_placeholder is None:
        # Get device from environment scene
        device = env.device if hasattr(env, 'device') else torch.device('cpu')
        # Return flattened image: batch_size * 480 * 640 * 3 elements
        _return_placeholder = torch.zeros((batch_size, 480 * 640 * 3), device=device)


    _camera_cache['frame_step'] = (_camera_cache['frame_step'] + 1) % max(1, _camera_cache['write_interval_steps'])


    scene_id = id(env.scene)
    if _camera_cache['last_scene_id'] != scene_id:
        _camera_cache['camera_keys'] = list(env.scene.keys())
        _camera_cache['available_cameras'] = [name for name in _camera_cache['camera_keys'] if "camera" in name.lower()]
        _camera_cache['last_scene_id'] = scene_id


    if _camera_cache['frame_step'] == 0:
        try:
            dt = getattr(env, 'physics_dt', 0.02)
            if hasattr(env.scene, 'sensors') and env.scene.sensors:
                for sensor in env.scene.sensors.values():
                    try:
                        sensor.update(dt, force_recompute=False)
                    except Exception:
                        pass
        except Exception:
            pass
    
    # Get the camera image directly as tensor
    camera_keys = _camera_cache['camera_keys']
    if "front_camera" in camera_keys:
        try:
            # Get tensor directly from camera data: [batch, height, width, 3]
            camera_images = env.scene["front_camera"].data.output["rgb"]
            
            # Ensure it's float and normalize to [0, 1]
            if camera_images.dtype != torch.float32:
                camera_images = camera_images.float()
            camera_images = camera_images / 255.0
            
            # Flatten each image in the batch: (batch, height, width, 3) -> (batch, height*width*3)
            flattened = camera_images.flatten(start_dim=1)
            
            # Ensure it's on the correct device
            device = _return_placeholder.device
            flattened = flattened.to(device=device, dtype=_return_placeholder.dtype)
            
            return flattened
            
        except Exception as e:
            print(f"[camera_state] Error processing camera image: {e}")
    
    # Return zero tensor if no camera found or error occurred
    return _return_placeholder

