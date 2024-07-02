import kornia
from kornia.feature import DISKFeatures
from cam_paramer.media_queue.FrameData import FrameData
from cam_paramer.media_queue.GroupData import GroupData
import torch
from typing import Tuple, Dict, List
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting tools

def lowes_ratio_filter(
    distances: torch.Tensor,
    matches: torch.Tensor,
    source_features: DISKFeatures,
    target_features: DISKFeatures
) -> Tuple[DISKFeatures, DISKFeatures]:
    """
    Apply Lowe's ratio test to filter matches between source and target features.

    Args:
    - distances (torch.Tensor): Distance tensor from matching process.
    - matches (torch.Tensor): Matches tensor from matching process.
    - source_features (DISKFeatures): Features of keypoints in the source image.
    - target_features (DISKFeatures): Features of keypoints in the target image.

    Returns:
    - Tuple[DISKFeatures, DISKFeatures]: Filtered source and target features after Lowe's ratio test.

    """
    good_matches = []
    distances = distances.flatten()

    for i in range(len(matches)):
        m, n = matches[i]
        if m < len(distances) and n < len(distances):
            if distances[m] < 0.75 * distances[n]:
                good_matches.append((m, n))

    if len(good_matches) < 5:
        print("Not enough good matches to compute the essential matrix")
        return source_features, target_features

    good_matches = torch.tensor(good_matches, dtype=torch.long)

    good_source_features = DISKFeatures(
        keypoints=source_features.keypoints[good_matches[:, 0]],
        descriptors=source_features.descriptors[good_matches[:, 0]],
        detection_scores=source_features.detection_scores[good_matches[:, 0]]
    )
    good_target_features = DISKFeatures(
        keypoints=target_features.keypoints[good_matches[:, 1]],
        descriptors=target_features.descriptors[good_matches[:, 1]],
        detection_scores=target_features.detection_scores[good_matches[:, 1]]
    )
    
    return good_source_features, good_target_features


def filter_inliers(
    source_features: DISKFeatures,
    target_features: DISKFeatures
) -> Tuple[DISKFeatures, DISKFeatures]:
    """
    Filter keypoints using RANSAC to estimate homography matrix.

    Args:
    - source_features (DISKFeatures): Features of keypoints in the source image.
    - target_features (DISKFeatures): Features of keypoints in the target image.

    Returns:
    - Tuple[DISKFeatures, DISKFeatures]: Filtered source and target features after RANSAC.

    Raises:
    - RuntimeError: If an error occurs during RANSAC estimation.

    """
    source_keypoints = source_features.keypoints.to(target_features.keypoints.device, target_features.keypoints.dtype)

    try:
        H, inliers = kornia.geometry.ransac.RANSAC(inl_th=2).forward(source_keypoints, target_features.keypoints, weights=None)
    except RuntimeError as e:
        print(f"Error during RANSAC: {e}")
        raise

    inliers_source_features = DISKFeatures(
        keypoints=source_features.keypoints[inliers.squeeze() == 1],
        descriptors=source_features.descriptors[inliers.squeeze() == 1],
        detection_scores=source_features.detection_scores[inliers.squeeze() == 1]
    )
    inliers_target_features = DISKFeatures(
        keypoints=target_features.keypoints[inliers.squeeze() == 1],
        descriptors=target_features.descriptors[inliers.squeeze() == 1],
        detection_scores=target_features.detection_scores[inliers.squeeze() == 1]
    )
    
    return inliers_source_features, inliers_target_features


def filter_line_matches_heatmap(
    source_matched_lines: torch.Tensor,
    source_output: Dict[str, torch.Tensor],
    target_matched_lines: torch.Tensor,
    target_output: Dict[str, torch.Tensor],
    confidence_threshold: float = 0.5
) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, List[torch.Tensor]]]:
    """
    Filter line matches based on heatmap confidence scores.

    Args:
    - source_matched_lines (torch.Tensor): Matched line segments from the source image.
    - source_output (Dict[str, torch.Tensor]): Output dictionary from source image processing.
    - target_matched_lines (torch.Tensor): Matched line segments from the target image.
    - target_output (Dict[str, torch.Tensor]): Output dictionary from target image processing.
    - confidence_threshold (float): Confidence threshold for filtering matches.

    Returns:
    - Tuple[Dict[str, List[torch.Tensor]], Dict[str, List[torch.Tensor]]]: Filtered line segments and keypoints dictionaries for source and target images.

    """
    source_line_heatmap = source_output["line_heatmap"].squeeze(0)
    source_junction_heatmap = source_output["junction_heatmap"].squeeze(0)
    target_line_heatmap = target_output["line_heatmap"].squeeze(0)
    target_junction_heatmap = target_output["junction_heatmap"].squeeze(0)
    source_lines, source_confidence, source_keypoints = [], [], []
    target_lines, target_confidence, target_keypoints = [], [], []

    for i, source_line_segment in enumerate(source_matched_lines):
        x1_s, y1_s, x2_s, y2_s = source_line_segment.flatten()
        confidence_s = (source_line_heatmap[int(y1_s), int(x1_s)] + source_line_heatmap[int(y2_s), int(x2_s)]) / 2.0
        junction_confidence_s = (source_junction_heatmap[int(y1_s), int(x1_s)] + source_junction_heatmap[int(y2_s), int(x2_s)]) / 2.0
        combined_confidence_s = (confidence_s + junction_confidence_s) / 2.0

        for j, target_line_segment in enumerate(target_matched_lines):
            x1_t, y1_t, x2_t, y2_t = target_line_segment.flatten()
            confidence_t = (target_line_heatmap[int(y1_t), int(x1_t)] + target_line_heatmap[int(y2_t), int(x2_t)]) / 2.0
            junction_confidence_t = (target_junction_heatmap[int(y1_t), int(x1_t)] + target_junction_heatmap[int(y2_t), int(x2_t)]) / 2.0
            combined_confidence_t = (confidence_t + junction_confidence_t) / 2.0

            if combined_confidence_s > confidence_threshold and combined_confidence_t > confidence_threshold:
                source_lines.append(source_line_segment)
                source_confidence.append(combined_confidence_s)
                source_keypoints.extend(source_line_segment)
                target_lines.append(target_line_segment)
                target_confidence.append(combined_confidence_t)
                target_keypoints.extend(target_line_segment)
                break

    if source_lines and target_lines:
        source_filtered_lines_dict = {"line_segments": source_lines, "keypoints": source_keypoints, "confidence": source_confidence}
        target_filtered_lines_dict = {"line_segments": target_lines, "keypoints": target_keypoints, "confidence": target_confidence}
        return source_filtered_lines_dict, target_filtered_lines_dict
    else:
        return None, None


def convert_lines_to_DiskFeatures(filtered_lines_dict: Dict[str, List[torch.Tensor]]) -> DISKFeatures:
    """
    Convert filtered line segments to DISKFeatures.

    Args:
    - filtered_lines_dict (Dict[str, List[torch.Tensor]]): Dictionary containing filtered line segments and their keypoints.

    Returns:
    - DISKFeatures: DISKFeatures object containing keypoints and confidence scores.

    Raises:
    - ValueError: If no valid lines are found in the input dictionary.

    """
    if len(filtered_lines_dict['line_segments']) > 0:
        keypoints = torch.stack(filtered_lines_dict["keypoints"])
        detection_scores = torch.tensor(filtered_lines_dict["confidence"], dtype=torch.float32)
        return DISKFeatures(keypoints=keypoints, detection_scores=detection_scores)
    else:
        raise ValueError(f"Cannot convert {len(filtered_lines_dict['line_segments'])} lines to DISKFeatures")


def split_frames_by_quality(group_data: GroupData, quality_threshold: int = 100) -> Tuple[List[FrameData], List[FrameData]]:
    """
    Split frames into high and low quality based on image quality scores.

    Args:
    - group_data (GroupData): Group data containing frame information.
    - quality_threshold (int): Threshold value for determining high quality frames.

    Returns:
    - Tuple[List[FrameData], List[FrameData]]: Lists of high and low quality frames.

    """
    hi_quality_frames = []
    lo_quality_frames = []
    frames = group_data.get_all_group_frame_data()

    for frame in frames:
        frame_quality = frame.image_quality
        if frame_quality >= quality_threshold:
            hi_quality_frames.append(frame)
        else:
            lo_quality_frames.append(frame)

    return hi_quality_frames, lo_quality_frames


def convert_lines_to_points(source_lines: torch.Tensor, target_lines: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert line segments to points.

    Args:
    - source_lines (torch.Tensor): Source line segments.
    - target_lines (torch.Tensor): Target line segments.

    Returns:
    - Tuple[torch.Tensor, torch.Tensor]: Source and target points.

    """
    source_points = source_lines.reshape(-1, 2)
    target_points = target_lines.reshape(-1, 2)
    return source_points, target_points


def convert_keypoints_to_3d(keypoints_2d: torch.Tensor, depthmap: torch.Tensor) -> torch.Tensor:
    """
    Convert 2D keypoints to 3D by concatenating with depth map.

    Args:
    - keypoints_2d (torch.Tensor): 2D keypoints.
    - depthmap (torch.Tensor): Depth map.

    Returns:
    - torch.Tensor: 3D keypoints.

    """
    keypoints_3d = torch.cat((keypoints_2d, depthmap), dim=1)
    return keypoints_3d


def visualize_keypoint_features(source_features: DISKFeatures, source_image: np.ndarray, target_features: DISKFeatures = None, target_image: np.ndarray = None, sec_delay: int = 0, scale_factor: float = 0.5):
    """
    Visualize 2D keypoints and their confidence scores on images.

    Args:
    - source_features (DISKFeatures): Features of keypoints in the source image.
    - source_image (np.ndarray): Source image.
    - target_features (DISKFeatures, optional): Features of keypoints in the target image.
    - target_image (np.ndarray, optional): Target image.
    - sec_delay (int): Delay in seconds before closing the image windows.
    - scale_factor (float): Scaling factor for display.

    """
    source_keypoints_2d = source_features.keypoints.clone().cpu().squeeze(0).numpy().astype(int)
    source_scores = source_features.detection_scores.clone().cpu().squeeze(0).numpy()

    if target_features is not None:
        target_keypoints_2d = target_features.keypoints.clone().cpu().squeeze(0).numpy().astype(int)
        target_scores = target_features.detection_scores.clone().cpu().squeeze(0).numpy()

    for idx, source_kp in enumerate(source_keypoints_2d):
        cv2.circle(source_image, (source_kp[0], source_kp[1]), radius=5, color=(0, 255, 0), thickness=-1)
        cv2.putText(source_image, f"{source_scores[idx]:.2f}", (source_kp[0], source_kp[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if target_image is not None and target_features is not None:
        for idx, target_kp in enumerate(target_keypoints_2d):
            cv2.circle(target_image, (target_kp[0], target_kp[1]), radius=5, color=(0, 0, 255), thickness=-1)
            cv2.putText(target_image, f"{target_scores[idx]:.2f}", (target_kp[0], target_kp[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        if source_image.shape[0] < source_image.shape[1]:
            display_image = np.vstack((source_image, target_image))
        else:
            display_image = np.hstack((source_image, target_image))
    else:
        display_image = source_image

    height, width = display_image.shape[:2]
    new_dimensions = (int(width * scale_factor), int(height * scale_factor))
    display_image = cv2.resize(display_image, new_dimensions, interpolation=cv2.INTER_AREA)

    cv2.imshow('Image 2D Keypoints', display_image)

    if sec_delay > 0:
        cv2.waitKey(sec_delay * 1000)
    else:
        cv2.waitKey(0)

    cv2.destroyAllWindows()


def visualize_line_segments(source_line_segments: torch.Tensor, source_image: np.ndarray, target_line_segments: torch.Tensor = None, target_image: np.ndarray = None, sec_delay: int = 0):
    """
    Visualize line segments on images.

    Args:
    - source_line_segments (torch.Tensor): Line segments in the source image.
    - source_image (np.ndarray): Source image.
    - target_line_segments (torch.Tensor, optional): Line segments in the target image.
    - target_image (np.ndarray, optional): Target image.
    - sec_delay (int): Delay in seconds before closing the image windows.

    """
    source_line_segments_np = source_line_segments.clone().cpu().numpy().astype(int)

    if target_line_segments is not None:
        target_line_segments_np = target_line_segments.clone().cpu().numpy().astype(int)

    for line in source_line_segments_np:
        cv2.line(source_image, (line[0, 0], line[0, 1]), (line[1, 0], line[1, 1]), color=(0, 255, 0), thickness=2)

    if target_image is not None and target_line_segments is not None:
        for line in target_line_segments_np:
            cv2.line(target_image, (line[0, 0], line[0, 1]), (line[1, 0], line[1, 1]), color=(0, 0, 255), thickness=2)

        display_image = np.hstack((source_image, target_image))
    else:
        display_image = source_image

    scale_factor = 0.5
    display_image = cv2.resize(display_image, (int(display_image.shape[1] * scale_factor), int(display_image.shape[0] * scale_factor)))

    cv2.imshow('Line Segments', display_image)

    if sec_delay > 0:
        cv2.waitKey(sec_delay * 1000)
    else:
        cv2.waitKey(0)

    cv2.destroyAllWindows()


def visualize_camera_positions(group_data: GroupData):
    """
    Visualize camera positions in 3D space.

    Args:
    - group_data (GroupData): Group data containing camera extrinsic parameters.

    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    all_frames = group_data.get_all_group_frame_data()
    camera_positions = [frame.extrinsic.camera_translation.cpu().numpy() for frame in all_frames if frame.camera_solved]

    for position in camera_positions:
        ax.scatter(position[0], position[1], position[2], marker='o', s=100)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_title('Camera Positions in 3D')

    plt.show()
