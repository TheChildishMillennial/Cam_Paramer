import torch
import os
from typing import List
from cam_paramer.media_queue.MediaData import MediaData
from cam_paramer.media_queue.FrameData import FrameData
from cam_paramer.media_queue.ImageData import ImageData
from cam_paramer.focal_length.focal_length import load_focal_length_model, CNN, FOCAL_LENGTH_MODEL_PATH
from cam_paramer.utils.system_utils import get_device
from cam_paramer.utils.output_utils import save_camera_extrinsic_data

class GroupData:
    """
    Class to manage a group of media data, including frames and images.
    """

    def __init__(self, group_dict: dict):
        """
        Initialize GroupData with a dictionary containing group information.

        Args:
        - group_dict (dict): Dictionary containing group_path, group_name, and media_paths.
        """
        self.media_group: List[MediaData] = []
        self.group_dir_path: str = group_dict['group_path']
        self.group_name: str = group_dict['group_name']
        self.group_size: int = 0
        
        # Load focal length model once and pass it into media_data to prevent multiple loads
        device: torch.device = get_device()
        focal_length_model: CNN = load_focal_length_model(FOCAL_LENGTH_MODEL_PATH, device)
        for media_path in group_dict['media_paths']:
            self.add_media_data(media_path, focal_length_model, device)
        del focal_length_model

    def get_media_data(self, item_idx: int) -> MediaData:
        """
        Get media data at a specific index.

        Args:
        - item_idx (int): Index of the media data.

        Returns:
        - MediaData: MediaData object at the specified index.
        """
        return self.media_group[item_idx]

    def remove_media_data(self, item_idx: int) -> None:
        """
        Remove media data at a specific index.

        Args:
        - item_idx (int): Index of the media data to remove.
        """
        self.media_group.pop(item_idx)
        self.group_size = len(self.media_group)

    def add_media_data(self, media_path: str, focal_length_model: CNN, device: torch.device) -> None:
        """
        Add media data to the group.

        Args:
        - media_path (str): Path to the media data.
        - focal_length_model (CNN): Pre-loaded focal length estimation model.
        - device (torch.device): Device on which the model should run.
        """
        media = MediaData(media_path, focal_length_model, device)
        self.media_group.append(media)
        self.group_size = len(self.media_group)
    
    def add_group_name(self, group_name: str) -> None:
        """
        Set the group name.

        Args:
        - group_name (str): Name to set for the group.
        """
        self.group_name = group_name

    def get_all_group_frame_data(self) -> List[FrameData]:  
        """
        Get all frame data from the entire media group.

        Returns:
        - List[FrameData]: List of all FrameData objects in the media group.
        """
        all_frames_data: List[FrameData] = []
        for media_data in self.media_group:
            all_frames_data.extend(media_data.frames)

        return all_frames_data
    
    def get_all_group_image_data(self) -> List[ImageData]:
        """
        Get all image data from the entire media group.

        Returns:
        - List[ImageData]: List of all ImageData objects in the media group.
        """
        all_images_data: List[ImageData] = []
        for media_data in self.media_group:
            for frame_data in media_data.frames:
                all_images_data.extend(frame_data.image)

        return all_images_data
    
    def scale_keypoints(self, keypoints_2d: torch.Tensor, original_height_px: int, original_width_px: int, scale_height_px: int, scale_width_px: int, device: torch.device='cpu', requires_grad: bool=False) -> torch.Tensor:
        """
        Scale 2D keypoints based on image scaling factors.

        Args:
        - keypoints_2d (torch.Tensor): 2D keypoints tensor.
        - original_height_px (int): Original height of the image in pixels.
        - original_width_px (int): Original width of the image in pixels.
        - scale_height_px (int): Target height for scaling in pixels.
        - scale_width_px (int): Target width for scaling in pixels.
        - device (torch.device, optional): Device on which to perform calculations (default: 'cpu').
        - requires_grad (bool, optional): Whether gradients are required (default: False).

        Returns:
        - torch.Tensor: Scaled 2D keypoints tensor.
        """
        original_h = torch.tensor(original_height_px, dtype=torch.int, device=device, requires_grad=requires_grad)
        original_w = torch.tensor(original_width_px, dtype=torch.int, device=device, requires_grad=requires_grad)
        scale_h = torch.tensor(scale_height_px, dtype=torch.int, device=device, requires_grad=requires_grad)
        scale_w = torch.tensor(scale_width_px, dtype=torch.int, device=device, requires_grad=requires_grad)
        
        # Calculate scale factors
        scale_factor_h = original_h / scale_h
        scale_factor_w = original_w / scale_w

        # Scale the keypoints back to original image size
        keypoints_2d[:, :, 0] *= scale_factor_h
        keypoints_2d[:, :, 1] *= scale_factor_w

        return keypoints_2d
        
    def save_all_extrinsic_data_json(self, output_dir: str):
        """
        Save all extrinsic data as JSON files.

        Args:
        - output_dir (str): Directory path where JSON files will be saved.
        """
        for media_data in self.media_group:
            group_name = self.group_name
            save_dir = os.path.join(output_dir, group_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # Save intrinsic matrix JSON
            media_data.mean_intrinsic.save_intrinsic_matrix_3x3_json(f"intrinsic_data_{media_data.file_name}", save_dir)
            
            # Save extrinsic data for each frame as JSON
            for frame_data in media_data.frames:
                if frame_data.camera_solved:
                    frame_num = frame_data.frame_number
                    extrinsic_data = frame_data.extrinsic
                    success = save_camera_extrinsic_data(extrinsic_data, save_dir, frame_num)
                    if not success:
                        raise ValueError(f"ERROR! - Something went wrong while saving frame: {frame_data.frame_number} to {save_dir}")
