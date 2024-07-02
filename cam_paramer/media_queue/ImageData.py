import numpy as np
import cv2
import torch

from cam_paramer.utils.input_utils import get_image_quality
from cam_paramer.media_queue.ImageTensorData import ImageTensorData

class ImageData:
    """
    Class to handle image data processing and manipulation.
    """

    def __init__(self, frame: np.ndarray):
        """
        Initialize ImageData with a numpy array representing an image frame.

        Args:
        - frame (np.ndarray): Input image frame as a numpy array.
        """
        self.frame: np.ndarray = frame
        self.frame_height: int = frame.shape[0]
        self.frame_width: int = frame.shape[1]
        self.frame_color_channels = frame.shape[2] if len(frame.shape) == 3 else 1

    def normalize_mean_std(self, image_data: np.ndarray = None) -> 'ImageData':
        """
        Normalize the image data based on mean and standard deviation.

        Args:
        - image_data (np.ndarray, optional): Input image data to normalize. Defaults to None (uses self.frame).

        Returns:
        - ImageData: Normalized ImageData object.
        """
        if image_data is None:
            image_data = self.frame
        mean = np.mean(image_data)
        std = np.std(image_data)
        return ImageData((image_data - mean) / std)
    
    def normalize_per_color_channel(self, image_data: np.ndarray = None) -> 'ImageData':
        """
        Normalize the image data per color channel by dividing by 255.

        Args:
        - image_data (np.ndarray, optional): Input image data to normalize. Defaults to None (uses self.frame).

        Returns:
        - ImageData: Normalized ImageData object.
        """
        if image_data is None:
            image_data = self.frame
        if image_data.ndim == 3 and image_data.shape[2] == 3:  # Check if it's a color image
            normalized_image = np.zeros_like(image_data, dtype=np.float32)
            for channel in range(image_data.shape[2]):
                normalized_image[..., channel] = image_data[..., channel] / 255.0
            return ImageData(normalized_image)
        else:
            return ImageData(image_data / 255.0)  # For grayscale images or single-channel images

    def normalize_per_max_frame(self, image_data: np.ndarray = None) -> 'ImageData':
        """
        Normalize the image data by dividing each pixel value by the maximum pixel value.

        Args:
        - image_data (np.ndarray, optional): Input image data to normalize. Defaults to None (uses self.frame).

        Returns:
        - ImageData: Normalized ImageData object.
        """
        if image_data is None:
            image_data = self.frame        
        normalized_image = image_data / np.max(image_data)
        return ImageData(normalized_image)
    
    def normalize_per_max_color_value(self, image_data: np.ndarray = None) -> 'ImageData':
        """
        Normalize the image data by dividing each pixel value by 255.

        Args:
        - image_data (np.ndarray, optional): Input image data to normalize. Defaults to None (uses self.frame).

        Returns:
        - ImageData: Normalized ImageData object.
        """
        if image_data is None:
            image_data = self.frame        
        normalized_image = image_data / 255
        return ImageData(normalized_image)
    
    def reshape_frame_px(self, image_data: np.ndarray = None, target_height_px: int = None, target_width_px: int = None, maintain_ratio: bool = False, add_padding: bool = False) -> 'ImageData':
        """
        Reshape the image data to a specific pixel size.

        Args:
        - image_data (np.ndarray, optional): Input image data to reshape. Defaults to None (uses self.frame).
        - target_height_px (int, optional): Target height in pixels for reshaping.
        - target_width_px (int, optional): Target width in pixels for reshaping.
        - maintain_ratio (bool, optional): Whether to maintain the aspect ratio during resizing (default: False).
        - add_padding (bool, optional): Whether to add padding to make the image square (default: False).

        Returns:
        - ImageData: Reshaped ImageData object.
        """
        if image_data is None:
            image_data = self.frame

        original_height, original_width = image_data.shape[:2]

        # If maintain_ratio is True, calculate the new size maintaining the aspect ratio
        if maintain_ratio:
            if target_height_px is not None and target_width_px is not None:
                aspect_ratio = original_width / original_height
                if target_width_px / aspect_ratio <= target_height_px:
                    target_height_px = int(target_width_px / aspect_ratio)
                else:
                    target_width_px = int(target_height_px * aspect_ratio)
            elif target_height_px is not None:
                target_width_px = int(target_height_px * (original_width / original_height))
            elif target_width_px is not None:
                target_height_px = int(target_width_px / (original_width / original_height))
            else:
                raise ValueError("Either target_height_px or target_width_px must be provided")
        else:
            if target_height_px is None or target_width_px is None:
                raise ValueError("Both target_height_px and target_width_px must be provided when not maintaining the ratio")

        # Resize the image
        resized_image = cv2.resize(image_data, (target_width_px, target_height_px))

        # If add_padding is True, pad the resized image to make it a square
        if add_padding:
            max_dim = max(target_height_px, target_width_px)
            top = (max_dim - target_height_px) // 2
            bottom = max_dim - target_height_px - top
            left = (max_dim - target_width_px) // 2
            right = max_dim - target_width_px - left

            padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            return ImageData(padded_image)

        return ImageData(resized_image)
    
    def reshape_frame_pc(self, image_data: np.ndarray, target_height_pc: float = None, target_width_pc: float = None, maintain_ratio: bool = False, add_padding: bool = False) -> 'ImageData':
        """
        Reshape the image data to a specific percentage of its original size.

        Args:
        - image_data (np.ndarray): Input image data to reshape.
        - target_height_pc (float, optional): Target height as a percentage of the original height.
        - target_width_pc (float, optional): Target width as a percentage of the original width.
        - maintain_ratio (bool, optional): Whether to maintain the aspect ratio during resizing (default: False).
        - add_padding (bool, optional): Whether to add padding to make the image square (default: False).

        Returns:
        - ImageData: Reshaped ImageData object.
        """
        if image_data is None:
            image_data = self.frame

        original_height, original_width = image_data.shape[:2]

        if maintain_ratio:
            if target_height_pc is not None and target_width_pc is not None:
                aspect_ratio = original_width / original_height
                if target_width_pc / aspect_ratio <= target_height_pc:
                    target_height_pc = target_width_pc / aspect_ratio
                else:
                    target_width_pc = target_height_pc * aspect_ratio
            elif target_height_pc is not None:
                target_width_pc = target_height_pc * (original_width / original_height)
            elif target_width_pc is not None:
                target_height_pc = target_width_pc / (original_width / original_height)
            else:
                raise ValueError("Either target_height_pc or target_width_pc must be provided")
        else:
            if target_height_pc is None or target_width_pc is None:
                raise ValueError("Both target_height_pc and target_width_pc must be provided when not maintaining the ratio")

        # Convert percentages to pixel dimensions
        target_height_px = int(original_height * target_height_pc)
        target_width_px = int(original_width * target_width_pc)

        # Resize the image
        resized_image = cv2.resize(image_data, (target_width_px, target_height_px))

        # If add_padding is True, pad the resized image to make it a square
        if add_padding:
            max_dim = max(target_height_px, target_width_px)
            top = (max_dim - target_height_px) // 2
            bottom = max_dim - target_height_px - top
            left = (max_dim - target_width_px) // 2
            right = max_dim - target_width_px - left

            padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            return ImageData(padded_image)

        return ImageData(resized_image)
    
    def get_frame_quality(self) -> float:
        """
        Get the quality score of the image frame.

        Returns:
        - float: Image quality score.
        """
        image_quality = get_image_quality(self.frame)
        return image_quality
    
    def to_tensor(self, image_data: np.ndarray = None, dtype: torch.dtype = torch.float32, device: torch.device = 'cpu', requires_grad: bool = False) -> ImageTensorData:
        """
        Convert image data to a PyTorch tensor.

        Args:
        - image_data (np.ndarray, optional): Input image data to convert to tensor. Defaults to None (uses self.frame).
        - dtype (torch.dtype, optional): Data type of the tensor (default: torch.float32).
        - device (torch.device, optional): Device on which to create the tensor (default: 'cpu').
        - requires_grad (bool, optional): Whether the tensor requires gradients (default: False).

        Returns:
        - ImageTensorData: Image tensor data object.
        """
        if image_data is None:
            image_data = self.frame
        image_tensor: torch.Tensor = torch.tensor(data=image_data, dtype=dtype, device=device, requires_grad=requires_grad)
        return ImageTensorData(image_tensor)
    
    def visualize_frame(self, view_timer_sec: int = 0):
        """
        Visualize the image frame using OpenCV.

        Args:
        - view_timer_sec (int, optional): Time duration (in seconds) to display the frame. Defaults to 0 (waits for key press).
        """
        cv2.imshow("ImageData Visualizer", self.frame)
        cv2.waitKey(view_timer_sec)
        cv2.destroyWindow("ImageData Visualizer")
