import os
import json
from typing import Union
from cam_paramer.media_queue.ExtrinsicData import ExtrinsicData
import numpy as np

def save_camera_intrinsic_matrix(intrinsic_matrix: Union[list, np.ndarray], output_path: str, input_file_name: str) -> bool:
    """
    Save camera intrinsic matrix to a JSON file.

    Args:
    - intrinsic_matrix (Union[list, np.ndarray]): Camera intrinsic matrix.
    - output_path (str): Output directory path.
    - input_file_name (str): Input file name (used for output file name).

    Returns:
    bool: True if saving succeeds, False otherwise.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    file_path = os.path.join(output_path, f"cam_intrinsic_{input_file_name}.json")
    
    with open(file_path, 'w') as f:
        json.dump(intrinsic_matrix.tolist(), f, indent=4)

    success = os.path.exists(file_path)
    return success

def save_camera_extrinsic_data(extrinsic_data: ExtrinsicData, output_path: str, input_file_name: str) -> bool:
    """
    Save camera extrinsic data to a JSON file.

    Args:
    - extrinsic_data (ExtrinsicData): Object containing extrinsic data.
    - output_path (str): Output directory path.
    - input_file_name (str): Input file name (used for output file name).

    Returns:
    bool: True if saving succeeds, False otherwise.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    file_path = os.path.join(output_path, f"cam_extrinsic_frame_{input_file_name}.json")
    
    # Convert torch tensors to lists for JSON serialization
    extrinsic_dict = {
        'extrinsic_matrix': extrinsic_data.extrinsic_matrix.clone().detach().cpu().tolist(),
        'rotation_matrix': extrinsic_data.camera_rotation_matrix.clone().detach().cpu().tolist(),
        'rotation_euler_deg': extrinsic_data.camera_rotation_euler_deg.clone().detach().cpu().tolist(),
        'translation_vector': extrinsic_data.camera_translation.clone().detach().cpu().tolist()
    }
    
    with open(file_path, 'w') as f:
        json.dump(extrinsic_dict, f, indent=4)

    success = os.path.exists(file_path)
    return success
