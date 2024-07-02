import torch
from typing import Dict
from kornia.feature import DISKFeatures
from cam_paramer.tracking.keypoint_tracking import detect_disk_keypoints
from cam_paramer.tracking.line_tracking import detect_sold2_lines

class ImageTensorData:
    """
    Class to encapsulate operations on image tensors in PyTorch format.
    """

    def __init__(self, image_tensor: torch.Tensor):
        """
        Initialize ImageTensorData with a PyTorch tensor representing an image.

        Args:
        - image_tensor (torch.Tensor): Input image tensor in PyTorch format.
        """
        self.image_tensor: torch.Tensor = image_tensor
        self.original_height: int = self.image_tensor.shape[0]
        self.original_width: int = self.image_tensor.shape[1]
        self.original_hw: tuple = self.image_tensor.shape[:2]

    def get_shape(self) -> torch.Size:
        """
        Get the shape of the image tensor.

        Returns:
        - torch.Size: Shape of the image tensor.
        """
        return self.image_tensor.shape
    
    def get_grad(self):
        """
        Get the gradient of the image tensor.

        Returns:
        - torch.Tensor: Gradient of the image tensor.
        """
        return self.image_tensor.grad
    
    def get_grad_fn(self):
        """
        Get the gradient function of the image tensor.

        Returns:
        - torch.autograd.Function: Gradient function of the image tensor.
        """
        return self.image_tensor.grad_fn
    
    def get_device(self) -> torch.device:
        """
        Get the device on which the image tensor is located.

        Returns:
        - torch.device: Device of the image tensor.
        """
        return self.image_tensor.device
    
    def detect_DISK_keypoints(self, DISK_model: torch.nn.Module, image_tensor: torch.Tensor = None, device: torch.device = 'cpu') -> DISKFeatures:
        """
        Detect DISK keypoints using a specified DISK model.

        Args:
        - DISK_model (torch.nn.Module): DISK model for keypoints detection.
        - image_tensor (torch.Tensor, optional): Input image tensor to detect keypoints. Defaults to None (uses self.image_tensor).
        - device (torch.device, optional): Device on which to perform detection (default: 'cpu').

        Returns:
        - DISKFeatures: Detected DISK keypoints.
        """
        if image_tensor is None:
            image_tensor = self.image_tensor.to(device=device, dtype=torch.float32)
        features: DISKFeatures = detect_disk_keypoints(image_tensor=image_tensor.permute(2, 0, 1).unsqueeze(0), DISK_model=DISK_model)
        return features
    
    def detect_SOLD2_lines(self, SOLD2_model: torch.nn.Module, image_tensor: torch.Tensor = None, device: torch.device = 'cpu') -> Dict[str, torch.Tensor]:
        """
        Detect SOLD2 lines using a specified SOLD2 model.

        Args:
        - SOLD2_model (torch.nn.Module): SOLD2 model for line detection.
        - image_tensor (torch.Tensor, optional): Input image tensor to detect lines. Defaults to None (uses self.image_tensor).
        - device (torch.device, optional): Device on which to perform detection (default: 'cpu').

        Returns:
        - Dict[str, torch.Tensor]: Detected SOLD2 lines and related information.
        """
        if image_tensor is None:
            image_tensor = self.image_tensor.to(device=device, dtype=torch.float32)
        output: Dict[str, torch.Tensor] = detect_sold2_lines(image_tensor=image_tensor, SOLD2_model=SOLD2_model)
        return output
    
    # TODO: Implement scaling lines function
    def scale_lines(self):
        """
        Scale lines based on some criteria.
        """
        pass  # Placeholder for future implementation