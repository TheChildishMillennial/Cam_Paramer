import numpy as np
from cam_paramer.media_queue.ImageData import ImageData
from cam_paramer.media_queue.IntrinsicData import IntrinsicData
from cam_paramer.media_queue.ExtrinsicData import ExtrinsicData
from cam_paramer.utils.input_utils import get_image_quality

class FrameData:
    """
    Class to hold information and operations related to a single frame of data.
    """

    def __init__(self, frame: np.ndarray, frame_idx: int, intrinsic_data: IntrinsicData, extrinsic_data: ExtrinsicData):
        """
        Initialize FrameData with frame information, intrinsic and extrinsic data.

        Args:
        - frame (np.ndarray): Array representing the frame image.
        - frame_idx (int): Index of the frame.
        - intrinsic_data (IntrinsicData): Intrinsic parameters of the camera.
        - extrinsic_data (ExtrinsicData): Extrinsic parameters of the camera.
        """
        self.image: ImageData = ImageData(frame)
        self.frame_number: int = frame_idx + 1
        self.frame_idx: int = frame_idx     
        self.frame_height: int = frame.shape[0]
        self.frame_width: int = frame.shape[1]
        self.frame_color_channels: int = frame.shape[2]
        self.image_quality: float = get_image_quality(frame)
        self.intrinsic: IntrinsicData = intrinsic_data
        self.extrinsic: ExtrinsicData = extrinsic_data
        self.camera_solved: bool = False

    def visualize_frame(self, view_simer_sec: int = 0):
        """
        Visualize the frame for a given duration.

        Args:
        - view_simer_sec (int): Duration to view the frame (in seconds).
        """
        self.image.visualize_frame(view_simer_sec)

    def add_extrinsic(self, extrinsic_data: ExtrinsicData) -> None:
        """
        Add or update extrinsic data for the frame.

        Args:
        - extrinsic_data (ExtrinsicData): New extrinsic data to be added.
        """
        self.extrinsic = extrinsic_data

    def set_cam_solved(self) -> None:
        """Mark the camera as solved for this frame."""
        self.camera_solved = True
