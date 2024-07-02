import kornia
import kornia.feature as KF
import torch
from typing import Tuple, Dict

def load_SOLD2_model(device: torch.device) -> torch.nn.Module:
    """
    Load SOLD2 model for line detection and matching.

    Args:
    - device (torch.device): Device to load the model on.

    Returns:
    - torch.nn.Module: Loaded SOLD2 model.

    Raises:
    - ValueError: If the SOLD2 model fails to load.

    """
    sold2_model = KF.SOLD2(pretrained=True, config=None).eval().to(device)
    if sold2_model is not None:
        print(f"SOLD2 Line Detector/Matcher Loaded to Device: {device}")
        return sold2_model
    else:
        raise ValueError("SOLD2 Failed to Load")
    

def detect_sold2_lines(image_tensor: torch.Tensor, SOLD2_model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """
    Detect lines in an image tensor using the SOLD2 model.

    Args:
    - image_tensor (torch.Tensor): Input image tensor.
    - SOLD2_model (torch.nn.Module): SOLD2 model for line detection.

    Returns:
    - Dict[str, torch.Tensor]: Dictionary containing line segments and descriptors.

    """
    image_tensor = torch.permute(image_tensor, (2, 0, 1)).unsqueeze(0)
    
    # Resize image if dimensions are larger than 800x800
    if image_tensor.shape[2] > 800 or image_tensor.shape[3] > 800:
        image_tensor = kornia.geometry.resize(image_tensor, (800, 800))
    
    # Convert to grayscale if image has more than one channel
    if image_tensor.shape[1] > 1:
        image_tensor = kornia.color.rgb_to_grayscale(image_tensor)
    
    with torch.inference_mode():
        outputs = SOLD2_model(image_tensor)
    
    return outputs

def match_sold2_lines(
    source_output: Dict[str, torch.Tensor],
    target_output: Dict[str, torch.Tensor],
    SOLD2_model: torch.nn.Module
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Match lines between source and target outputs using the SOLD2 model.

    Args:
    - source_output (Dict[str, torch.Tensor]): Output dictionary from source image.
    - target_output (Dict[str, torch.Tensor]): Output dictionary from target image.
    - SOLD2_model (torch.nn.Module): SOLD2 model for line matching.

    Returns:
    - Tuple[torch.Tensor, torch.Tensor]: Matched source and target line segments.

    """
    source_line_segments = source_output['line_segments'][0]
    target_line_segments = target_output['line_segments'][0]
    source_descriptors = source_output['dense_desc'][0].unsqueeze(0)
    target_descriptors = target_output['dense_desc'][0].unsqueeze(0)
    
    with torch.inference_mode():
        matches = SOLD2_model.match(source_line_segments, target_line_segments, source_descriptors, target_descriptors)
    
    valid_matches = matches != -1
    match_indices = matches[valid_matches]
    source_matched_lines = source_line_segments[valid_matches]
    target_matched_lines = target_line_segments[match_indices]
    
    return source_matched_lines, target_matched_lines
