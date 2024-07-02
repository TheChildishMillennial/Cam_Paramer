import torch
from torch.nn import Module
from cam_paramer.tracking.keypoint_tracking import load_DISK_model, load_LightGlue_model, match_keypoints_lightglue
from cam_paramer.tracking.line_tracking import load_SOLD2_model, match_sold2_lines
from cam_paramer.tracking.tracking_utils import lowes_ratio_filter, filter_inliers, filter_line_matches_heatmap, convert_lines_to_DiskFeatures, split_frames_by_quality
from cam_paramer.media_queue.FramePairManager import FramePairManager
from cam_paramer.media_queue.GroupData import GroupData
from cam_paramer.media_queue.FrameData import FrameData
from cam_paramer.media_queue.MediaData import MediaData
from cam_paramer.media_queue.ImageTensorData import ImageTensorData
import numpy as np
from kornia.feature import DISKFeatures
from typing import Dict, Tuple, List, Optional
import cv2

def detect_motion_from_keypoints(
    source_image_tensor_data: ImageTensorData,
    target_image_tensor_data: ImageTensorData,
    source_hw: torch.Tensor,
    target_hw: torch.Tensor,
    DISK_model: Module,
    LightGlue_model: Module,
    device: torch.device
) -> Tuple[DISKFeatures, DISKFeatures]:
    """
    Detects motion between two images using keypoints and matching.

    Args:
    - source_image_tensor_data: Tensor data of the source image.
    - target_image_tensor_data: Tensor data of the target image.
    - source_hw: Tensor containing height and width of the source image.
    - target_hw: Tensor containing height and width of the target image.
    - DISK_model: PyTorch module for DISK keypoint detection.
    - LightGlue_model: PyTorch module for LightGlue keypoint matching.
    - device: Torch device for computation.

    Returns:
    - Tuple of DISKFeatures for source and target images.
    """
    print("Detecting Keypoints With DISK Model")
    source_features = source_image_tensor_data.detect_DISK_keypoints(DISK_model=DISK_model, image_tensor=source_image_tensor_data.image_tensor, device=device)
    target_features = target_image_tensor_data.detect_DISK_keypoints(DISK_model=DISK_model, image_tensor=target_image_tensor_data.image_tensor, device=device)
    print(f"Matching {len(source_features.keypoints)} Keypoints With DISK Model")
    
    distances, matches = match_keypoints_lightglue(
        LightGlue_model,
        source_features,
        target_features,
        source_hw.tolist(),
        target_hw.tolist(),
        device        
    )
    
    print(f"Filtering {len(source_features.keypoints)} Found Keypoints")
    source_features, target_features = lowes_ratio_filter(distances, matches, source_features, target_features)
    print(f"{len(source_features.keypoints)} Keypoints Passed Lowes Ratio Filter")
    
    source_features, target_features = filter_inliers(source_features, target_features)
    print(f"{len(source_features.keypoints)} Keypoints Passed Inliers Filter")
    
    if torch.numel(source_features.keypoints) != torch.numel(target_features.keypoints) or torch.numel(source_features.keypoints) < 5:
        print(f"WARNING - 5 Keypoints Required to Solve. Only {len(source_features.keypoints)} Quality Keypoints Were Detected")
        return None, None
    else:
        print(f"{len(source_features.keypoints)} Quality Keypoints Were Detected")
        return source_features, target_features

def detect_motion_from_lines(
    source_image_tensor_data: ImageTensorData,
    target_image_tensor_data: ImageTensorData,
    SOLD2_model: Module,
    device: torch.device
) -> Tuple[DISKFeatures, DISKFeatures]:
    """
    Detects motion between two images using lines detected by SOLD2 model.

    Args:
    - source_image_tensor_data: Tensor data of the source image.
    - target_image_tensor_data: Tensor data of the target image.
    - SOLD2_model: PyTorch module for SOLD2 line detection.
    - device: Torch device for computation.

    Returns:
    - Tuple of DISKFeatures for source and target images.
    """
    print(f"Attempting to track lines with SOLD2")
    
    # Detect lines with SOLD2 model
    source_output: Dict[str, torch.Tensor] = source_image_tensor_data.detect_SOLD2_lines(SOLD2_model=SOLD2_model, image_tensor=source_image_tensor_data.image_tensor, device=device)
    target_output: Dict[str, torch.Tensor] = target_image_tensor_data.detect_SOLD2_lines(SOLD2_model=SOLD2_model, image_tensor=target_image_tensor_data.image_tensor, device=device)
    
    # Match lines between source and target outputs
    source_lines, target_lines = match_sold2_lines(source_output=source_output, target_output=target_output, SOLD2_model=SOLD2_model)
    print(f"Found Source: {len(source_lines)} Lines, Target: {len(target_lines)} Lines")
    print(f"Filtering Found Line Matches")
    source_filtered_lines_dict, target_filtered_lines_dict = filter_line_matches_heatmap(source_lines, source_output, target_lines, target_output)
    
    if source_filtered_lines_dict is not None and target_filtered_lines_dict is not None:
        print(f"{len(source_filtered_lines_dict['line_segments'])} Lines Passed Heatmap Filter")
        if len(source_filtered_lines_dict['keypoints']) > 8:
            source_features: DISKFeatures = convert_lines_to_DiskFeatures(source_filtered_lines_dict)
            target_features: DISKFeatures = convert_lines_to_DiskFeatures(target_filtered_lines_dict)
            print(f"{len(source_features.keypoints)} Quality Keypoints Were Detected")
            return source_features, target_features
        else:
            print(f"WARNING - 8 Keypoints Required to Solve. Only {len(source_filtered_lines_dict['keypoints'])} Quality Keypoints Were Detected")
            return None, None
    else:
        print(f"WARNING - No Quality Lines Were Detected")
        return None, None

def interpolate_motion_from_video(media_data: MediaData):
    """
    Interpolates camera motion between solved frames in a video.

    Args:
    - media_data: MediaData object containing video frames and associated data.

    Raises:
    - ValueError: If the media type is not compatible (e.g., input is not a video).

    Prints:
    - Interpolates camera motion for each frame gap between solved frames.
    - Prints the number of unsolved frames and their indices.

    """
    if media_data.media_type == "image":
        raise ValueError(f"Media: {media_data.file_path} is of Media Type: {media_data.media_type} and is not compatible with interpolation")
    
    print(f"Interpolating {media_data.file_path}")
    
    last_solved_frame: FrameData = None
    next_solved_frame: FrameData = None
    frames_to_interpolate: List[FrameData] = []
    current_idx = 0

    while current_idx < len(media_data.frames):
        frame = media_data.frames[current_idx]
        
        # Check if the current frame is solved
        if frame.camera_solved:
            next_idx = current_idx + 1
            
            if next_idx < len(media_data.frames):
                # Check if the next frame is solved
                if media_data.frames[next_idx].camera_solved:
                    last_solved_frame = None
                    next_solved_frame = None
                else:
                    last_solved_frame = frame
                    is_searching = True
                    search_stride = 1
                    frames_to_interpolate = []

                    while is_searching and (current_idx + search_stride) < len(media_data.frames):
                        search_idx = current_idx + search_stride
                        search_frame = media_data.frames[search_idx]
                        
                        if not search_frame.camera_solved:
                            frames_to_interpolate.append(search_frame)
                        else:
                            next_solved_frame = search_frame
                            is_searching = False

                        search_stride += 1

                    if last_solved_frame is not None and next_solved_frame is not None:
                        frame_gap = len(frames_to_interpolate)
                        rotation_gap = next_solved_frame.extrinsic.camera_rotation_matrix - last_solved_frame.extrinsic.camera_rotation_matrix
                        translation_gap = next_solved_frame.extrinsic.camera_translation - last_solved_frame.extrinsic.camera_translation
                        rotation_unit = rotation_gap / (frame_gap + 1)
                        translation_unit = translation_gap / (frame_gap + 1)

                        for i in range(frame_gap):
                            interpolated_rotation = last_solved_frame.extrinsic.camera_rotation_matrix + (rotation_unit * (i + 1))
                            interpolated_translation = last_solved_frame.extrinsic.camera_translation + (translation_unit * (i + 1))
                            frames_to_interpolate[i].extrinsic.add_camera_rotation_matrix(interpolated_rotation)
                            frames_to_interpolate[i].extrinsic.add_camera_translation(interpolated_translation)
                            frames_to_interpolate[i].set_cam_solved()
                        
                        last_solved_frame = None
                        next_solved_frame = None
                        frames_to_interpolate = []

        current_idx += 1

    unsolved_frames = [frame for frame in media_data.frames if not frame.camera_solved]
    if unsolved_frames:
        print(f"Unsolved Frames: {len(unsolved_frames)}")
        for frame in unsolved_frames:
            print(f"Unsolved Frame Index: {frame.frame_idx}")


def solve_camera_motion(source_features: DISKFeatures, target_features: DISKFeatures, source_frame: FrameData, target_frame: FrameData, frame_pair_manager: FramePairManager, device: torch.device) -> bool:
    """
    Solve camera motion between two frames using DISKFeatures and essential matrix computation.

    Args:
    - source_features: DISKFeatures object containing keypoints and scores from the source frame.
    - target_features: DISKFeatures object containing keypoints and scores from the target frame.
    - source_frame: FrameData object representing the source frame data.
    - target_frame: FrameData object representing the target frame data.
    - frame_pair_manager: FramePairManager object managing frame pairs and solving status.
    - device: torch.device specifying the device for tensor operations.

    Returns:
    - bool: True if the camera motion was successfully solved and updated, False otherwise.

    Raises:
    - RuntimeError: If the essential matrix computation fails or returns an empty result.

    """
    # Compute normalized weights
    normalized_weights = torch.min(source_features.detection_scores, target_features.detection_scores)

    # Ensure weights are in the correct shape
    weight_mask = (normalized_weights > 0).float().cpu().numpy().astype(np.uint8).squeeze()

    # Compute the essential matrix
    source_keypoints = source_features.keypoints.clone().detach().cpu().numpy()
    target_keypoints = target_features.keypoints.clone().detach().cpu().numpy()

    # Ensure weight_mask has the correct shape
    if weight_mask.ndim == 1:
        weight_mask = weight_mask[:, np.newaxis]

    E_mat, _ = cv2.findEssentialMat(source_keypoints, target_keypoints, source_frame.intrinsic.intrinsic_matrix, mask=weight_mask)
    if E_mat is None or E_mat.size == 0:
        raise RuntimeError("Failed to compute essential matrix")

    # Recover pose from the essential matrix
    _, rotation, translation, mask = cv2.recoverPose(E_mat, source_keypoints, target_keypoints, source_frame.intrinsic.intrinsic_matrix, mask=weight_mask)

    # Convert rotation and translation back to PyTorch tensors
    rotation_tensor = torch.tensor(rotation, dtype=torch.float32, device=device, requires_grad=True)
    translation_tensor = torch.tensor(translation, dtype=torch.float32, device=device, requires_grad=True).squeeze(0)

    # Compute combined rotation and translation
    combined_rotation = source_frame.extrinsic.camera_rotation_matrix @ rotation_tensor
    combined_translation = source_frame.extrinsic.camera_translation + translation_tensor

    # Update target frame extrinsics
    target_frame.extrinsic.add_camera_rotation_matrix(combined_rotation)
    target_frame.extrinsic.add_camera_translation(combined_translation)
    target_frame.set_cam_solved()

    # Check if the frame pair manager was solved
    return frame_pair_manager.was_solved()

def track_camera_motion(group_data: GroupData, device: str = 'cpu', frame_quality_threshold: Optional[int] = None, interpolate_video_motion: bool = True):
    """
    Track camera motion across frames in a group of media data.

    Args:
    - group_data (GroupData): Object containing frames grouped by media.
    - device (str, optional): Device for tensor operations (default is 'cpu').
    - frame_quality_threshold (int, optional): Threshold to classify frames as quality or low-quality (default is None).
    - interpolate_video_motion (bool, optional): Flag to interpolate video motion if True (default is True).

    Returns:
    - None

    """
    quality_frames, low_quality_frames = split_frames_by_quality(group_data, frame_quality_threshold)
    print(f"Quality Frames: {len(quality_frames)}")
    
    DISK_model = load_DISK_model(device=device)
    LightGlue_model = load_LightGlue_model(device=device)
    SOLD2_model = None
    frame_pair_manager = FramePairManager(quality_frames)
    is_solving = True
    
    while is_solving:
        print(f"Is Solving: {is_solving}")
        source_frame, target_frame = frame_pair_manager.get_frame_pair()
        
        if source_frame.frame_idx == 0:
            source_frame.set_cam_solved()
        
        source_image_tensor = ImageTensorData(source_frame.image.to_tensor(device=device, requires_grad=True))
        target_image_tensor = ImageTensorData(target_frame.image.to_tensor(device=device, requires_grad=True))
        
        source_hw = (source_frame.frame_height, source_frame.frame_width)
        target_hw = (target_frame.frame_height, target_frame.frame_width)
        
        source_hw_tensor = torch.tensor(source_hw, device=device)
        target_hw_tensor = torch.tensor(target_hw, device=device)
        
        source_features, target_features = detect_motion_from_keypoints(
            source_image_tensor, target_image_tensor,
            source_hw_tensor, target_hw_tensor, DISK_model, LightGlue_model, device
        )
        
        if source_features is None and target_features is None or len(source_features.keypoints) < 8 and len(target_features.keypoints) < 8:
            if SOLD2_model is None:
                SOLD2_model = load_SOLD2_model(device=device)
            
            source_features, target_features = detect_motion_from_lines(
                source_image_tensor, target_image_tensor,
                SOLD2_model, device
            )
            
            if source_features is None and target_features is None or len(source_features.keypoints) < 8 and len(target_features.keypoints) < 8:
                if frame_pair_manager.was_unsolved():
                    is_solving = False
                    break
                else:
                    continue
            else:
                if solve_camera_motion(source_features, target_features, source_frame, target_frame, frame_pair_manager, device):
                    is_solving = False
        else:
            if solve_camera_motion(source_features, target_features, source_frame, target_frame, frame_pair_manager, device):
                is_solving = False
    
    if interpolate_video_motion:
        for media_data in group_data.media_groups:
            if media_data.media_type == "video":
                interpolate_motion_from_video(media_data)
