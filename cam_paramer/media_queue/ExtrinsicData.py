import torch
from cam_paramer.utils.cam_utils import rotation_matrix_to_euler_angles, euler_angles_to_rotation_matrix

class ExtrinsicData:
    """
    Class to manage extrinsic camera data including rotation and translation.
    """

    def __init__(self, device: torch.device):
        """
        Initialize ExtrinsicData with tensors for rotation, translation, and extrinsic matrix.
        
        Args:
        - device (torch.device): Device to store tensors (e.g., 'cpu', 'cuda').
        """
        self.camera_rotation_euler_deg: torch.Tensor = torch.zeros(3, 1, dtype=torch.float32, device=device, requires_grad=True)
        self.camera_rotation_matrix: torch.Tensor = torch.eye(3, dtype=torch.float32, device=device, requires_grad=True)
        self.camera_translation: torch.Tensor = torch.zeros(3, 1, dtype=torch.float32, device=device, requires_grad=True)
        self.extrinsic_matrix: torch.Tensor = torch.eye(4, dtype=torch.float32, device=device, requires_grad=True)

    def add_camera_rotation_matrix(self, rotation_matrix: torch.Tensor) -> None:
        """
        Set camera rotation matrix and update extrinsic matrix.

        Args:
        - rotation_matrix (torch.Tensor): 3x3 rotation matrix.
        """
        self.camera_rotation_matrix = rotation_matrix
        self.extrinsic_matrix[:3, :3] = self.camera_rotation_matrix
        rotation_euler_deg = rotation_matrix_to_euler_angles(self.camera_rotation_matrix)
        self.camera_rotation_euler_deg = torch.tensor(rotation_euler_deg, dtype=torch.float32, device=rotation_matrix.device).unsqueeze(1)

    def add_camera_rotation_euler_deg(self, euler_rotation_deg: torch.Tensor) -> None:
        """
        Set camera rotation angles in degrees and update rotation matrix and extrinsic matrix.

        Args:
        - euler_rotation_deg (torch.Tensor): Euler angles (in degrees) as 3x1 tensor.
        """
        self.camera_rotation_euler_deg = euler_rotation_deg.unsqueeze(1)  # Ensure the format matches
        rotation_matrix = euler_angles_to_rotation_matrix(self.camera_rotation_euler_deg.squeeze(1))
        self.camera_rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float32, device=euler_rotation_deg.device)
        self.extrinsic_matrix[:3, :3] = self.camera_rotation_matrix

    def add_camera_translation(self, camera_translation: torch.Tensor) -> None:
        """
        Set camera translation vector and update extrinsic matrix.

        Args:
        - camera_translation (torch.Tensor): 3x1 translation vector.
        """
        camera_translation = camera_translation.squeeze()  # Ensure camera_translation is 1D
        self.camera_translation = camera_translation.unsqueeze(1)  # Retain the original format
        self.extrinsic_matrix[:3, 3] = camera_translation

    def get_camera_translation_meters(self, pixel_size_mm):
        """
        Convert camera translation from pixel units to meters.

        Args:
        - pixel_size_mm (float): Pixel size in millimeters.

        Returns:
        - torch.Tensor: Camera translation in meters.
        """
        pixel_size_m = pixel_size_mm / 1000.0  # Convert mm to meters
        camera_translation_meters = self.camera_translation * pixel_size_m
        return camera_translation_meters
