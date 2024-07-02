from kornia.feature import DISKFeatures
import kornia.feature as KF
import torch
from torch.nn import Module
from typing import Tuple

def load_LightGlue_model(device: torch.device='cpu', feature_detector: str="disk") -> torch.nn.Module:
    """
    Load LightGlue model for keypoint matching.

    Args:
    - device (torch.device, optional): Device to load the model on (default is 'cpu').
    - feature_detector (str, optional): Feature detector type (default is 'disk').

    Returns:
    - torch.nn.Module: Loaded LightGlue model.

    Raises:
    - ValueError: If the LightGlue matcher fails to load.

    """
    lg_matcher = None
    print(f"Attempting to Load LightGlue to {device}")
    lg_matcher = KF.LightGlueMatcher(feature_detector).eval().to(device)
    if lg_matcher is not None:
        print(f"LightGlue Keypoint Matcher Loaded to Device: {device}")
        return lg_matcher
    else:
        raise ValueError("LightGlue Matcher Failed to Load")
    

def load_DISK_model(device: torch.device='cpu') -> torch.nn.Module:
    """
    Load DISK model for keypoint detection.

    Args:
    - device (torch.device, optional): Device to load the model on (default is 'cpu').

    Returns:
    - torch.nn.Module: Loaded DISK model.

    Raises:
    - ValueError: If the DISK model fails to load.

    """
    disk = None
    print(f"Attempting to Load DISK to {device}")
    disk = KF.DISK.from_pretrained("depth").to(device)
    if disk is not None:
        print(f"DISK Keypoint Detector Loaded to Device: {device}")
        return disk
    else:
        raise ValueError("DISK Failed to Load")
    

def detect_disk_keypoints(
    image_tensor: torch.Tensor,
    DISK_model: torch.nn.Module,
    max_features: int=2048
) -> DISKFeatures:
    """
    Detect DISK keypoints from an input image tensor using a DISK model.

    Args:
    - image_tensor (torch.Tensor): Input image tensor.
    - DISK_model (torch.nn.Module): DISK model for keypoints detection.
    - max_features (int, optional): Maximum number of keypoints to detect (default is 2048).

    Returns:
    - DISKFeatures: Detected DISK keypoints and descriptors.

    """
    with torch.inference_mode():
        features_list = DISK_model(image_tensor, max_features, pad_if_not_divisible=True)
        features: DISKFeatures = features_list[0]
    return features


def match_keypoints_lightglue(
    LightGlue_model: Module,
    source_features: DISKFeatures,
    target_features: DISKFeatures,
    source_image_hw: Tuple[int, int] = None,
    target_image_hw: Tuple[int, int] = None,
    device: torch.device = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Match keypoints between source and target DISKFeatures using LightGlue model.

    Args:
    - LightGlue_model (Module): LightGlue model for matching keypoints.
    - source_features (DISKFeatures): Source DISKFeatures containing keypoints and descriptors.
    - target_features (DISKFeatures): Target DISKFeatures containing keypoints and descriptors.
    - source_image_hw (Tuple[int, int], optional): Source image height and width (default is None).
    - target_image_hw (Tuple[int, int], optional): Target image height and width (default is None).
    - device (torch.device, optional): Device for tensor operations (default is 'cpu').

    Returns:
    - Tuple[torch.Tensor, torch.Tensor]: Distances and matches between keypoints.

    """
    source_lafs = KF.laf_from_center_scale_ori(
        source_features.keypoints.unsqueeze(0),
        torch.ones(1, len(source_features.keypoints), 1, 1, device=device)
    )
    
    target_lafs = KF.laf_from_center_scale_ori(
        target_features.keypoints.unsqueeze(0),
        torch.ones(1, len(target_features.keypoints), 1, 1, device=device)
    )

    if source_image_hw is not None:
        source_image_hw = tuple(torch.tensor(x, device=device) for x in source_image_hw)
    if target_image_hw is not None:
        target_image_hw = tuple(torch.tensor(x, device=device) for x in target_image_hw)

    distances, matches = LightGlue_model(
        source_features.descriptors.clone().detach(),
        target_features.descriptors.clone().detach(),
        source_lafs.clone().detach(),
        target_lafs.clone().detach(),
        hw1=source_image_hw,
        hw2=target_image_hw
    )
    
    return distances, matches
