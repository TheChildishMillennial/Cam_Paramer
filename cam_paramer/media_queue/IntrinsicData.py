import numpy as np
import json
import os
from cam_paramer.utils.cam_utils import scale_focal_length_to_pixel, create_3x3_intrinsic_matrix

class IntrinsicData:
    """
    Class to manage intrinsic camera parameters, including focal length and intrinsic matrices.
    """

    def __init__(self, focal_length_mm: float, pixel_size_mm: float, frame_height_px: int, frame_width_px: int, intrinsic_matrix: np.ndarray = None):
        """
        Initialize IntrinsicData with focal length, pixel size, frame height, frame width, and optional intrinsic matrix.

        Args:
        - focal_length_mm (float): Focal length in millimeters.
        - pixel_size_mm (float): Pixel size in millimeters.
        - frame_height_px (int): Frame height in pixels.
        - frame_width_px (int): Frame width in pixels.
        - intrinsic_matrix (np.ndarray, optional): Initial intrinsic matrix. Defaults to None.
        """
        self.focal_length_mm: float = focal_length_mm
        self.focal_length_px: int = scale_focal_length_to_pixel(self.focal_length_mm, pixel_size_mm)
        
        if intrinsic_matrix is None:
            self.intrinsic_matrix = create_3x3_intrinsic_matrix(self.focal_length_px, frame_height_px, frame_width_px)
        else:
            self.intrinsic_matrix = intrinsic_matrix

    def get_4x4_intrinsic_matrix(self) -> np.ndarray:
        """
        Get the intrinsic matrix as a 4x4 matrix with the upper-left 3x3 part being the intrinsic matrix.

        Returns:
        - np.ndarray: 4x4 intrinsic matrix.
        """
        intrinsic_4x4 = np.eye(4)
        intrinsic_4x4[:3, :3] = self.intrinsic_matrix
        return intrinsic_4x4
    
    def save_intrinsic_matrix_3x3_json(self, save_as_name: str, output_dir: str):
        """
        Save the 3x3 intrinsic matrix as a JSON file.

        Args:
        - save_as_name (str): File name to save as (without extension).
        - output_dir (str): Directory to save the JSON file.
        """
        save_dir = os.path.join(output_dir, f"{save_as_name}.json")
        int_mat = self.intrinsic_matrix.tolist()
        with open(save_dir, 'w') as json_file:
            json.dump(int_mat, json_file, indent=1)

    def save_intrinsic_matrix_3x3_npy(self, save_as_name: str, output_dir: str):
        """
        Save the 3x3 intrinsic matrix as a NumPy .npy file.

        Args:
        - save_as_name (str): File name to save as (without extension).
        - output_dir (str): Directory to save the .npy file.
        """
        save_path = os.path.join(output_dir, f"{save_as_name}.npy")
        np.save(save_path, self.intrinsic_matrix)

    def save_intrinsic_matrix_3x3_txt(self, save_as_name: str, output_dir: str):
        """
        Save the 3x3 intrinsic matrix as a text file (.txt).

        Args:
        - save_as_name (str): File name to save as (without extension).
        - output_dir (str): Directory to save the .txt file.
        """
        save_path = os.path.join(output_dir, f"{save_as_name}.txt")
        np.savetxt(save_path, self.intrinsic_matrix, fmt='%f')
