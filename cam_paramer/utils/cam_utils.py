import numpy as np
from typing import Tuple
import math
import kornia
import torch

def scale_focal_length_to_pixel(focal_length_mm: float, pixel_size_mm: float) -> float:
    """
    Convert focal length from millimeters to pixels.

    Args:
    - focal_length_mm (float): Focal length in millimeters.
    - pixel_size_mm (float): Pixel size in millimeters.

    Returns:
    - float: Focal length in pixels.

    """
    focal_length_px = focal_length_mm / pixel_size_mm
    return focal_length_px

def create_3x3_intrinsic_matrix(focal_length_px: float, frame_height: int, frame_width: int) -> np.ndarray:
    """
    Create a 3x3 intrinsic matrix.

    Args:
    - focal_length_px (float): Focal length in pixels.
    - frame_height (int): Height of the image frame.
    - frame_width (int): Width of the image frame.

    Returns:
    - np.ndarray: 3x3 intrinsic matrix.

    """
    intrinsic_matrix = np.eye(3)
    cx = frame_width // 2
    cy = frame_height // 2
    intrinsic_matrix[0, 0] = focal_length_px
    intrinsic_matrix[1, 1] = focal_length_px
    intrinsic_matrix[0, 2] = cx
    intrinsic_matrix[1, 2] = cy
    return intrinsic_matrix

def convert_3x3_intrinsic_to_4x4(intrinsic_matrix: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 intrinsic matrix to a 4x4 matrix.

    Args:
    - intrinsic_matrix (np.ndarray): 3x3 intrinsic matrix.

    Returns:
    - np.ndarray: 4x4 intrinsic matrix.

    """
    intrinsic = np.eye(4)
    intrinsic[:3, :3] = intrinsic_matrix
    return intrinsic

def rotation_matrix_to_euler_angles(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert a rotation matrix to Euler angles (XYZ convention).

    Args:
    - R (np.ndarray): 3x3 rotation matrix.

    Returns:
    - Tuple[float, float, float]: Euler angles (phi, theta, psi) in radians.

    """
    assert R.shape == (3, 3), "The rotation matrix must be 3x3"

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return x, y, z

def euler_angles_to_rotation_matrix(euler_angles: np.ndarray) -> np.ndarray:
    """
    Convert Euler angles (XYZ convention) to a rotation matrix.

    Args:
    - euler_angles (np.ndarray): Euler angles (phi, theta, psi) in radians.

    Returns:
    - np.ndarray: 3x3 rotation matrix.

    """
    assert euler_angles.shape == (3,), "Euler angles must be a 3-element array"

    phi, theta, psi = euler_angles

    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)]
    ])

    R_y = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    R_z = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
    ])

    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def estimate_camera_pose_from_keypoints(source_keypoints: torch.Tensor, target_keypoints: torch.Tensor) -> torch.Tensor:
    """
    Estimate camera pose from keypoints using the 5-point algorithm.

    Args:
    - source_keypoints (torch.Tensor): Source keypoints tensor.
    - target_keypoints (torch.Tensor): Target keypoints tensor.

    Returns:
    - torch.Tensor: Essential matrix estimated from keypoints.

    """
    E_mat = kornia.geometry.find_essential(source_keypoints.unsqueeze(0), target_keypoints.unsqueeze(0)).squeeze(0)
    return E_mat
