from typing import List, Tuple
import torch
import cv2
import numpy as np
from tqdm import tqdm

from cam_paramer.focal_length.focal_length import CNN, estimate_focal_length
from cam_paramer.utils.input_utils import split_file_name, get_media_type, get_dpi, get_pixel_size_mm
from cam_paramer.utils.system_utils import get_device
from cam_paramer.utils.video_utils import read_video, get_total_video_frames, get_video_fps, get_video_frame, close_video
from cam_paramer.utils.cam_utils import create_3x3_intrinsic_matrix
from cam_paramer.media_queue.FrameData import FrameData
from cam_paramer.media_queue.IntrinsicData import IntrinsicData
from cam_paramer.media_queue.ExtrinsicData import ExtrinsicData

class MediaData:
    """
    Class for managing media files (images or videos), processing frames, estimating focal lengths,
    and managing intrinsic and extrinsic camera parameters.
    """

    def __init__(self, file_path: str, focal_length_model: CNN, device: torch.device):
        """
        Initialize MediaData with file path, focal length model, and device.

        Args:
        - file_path (str): Path to the media file.
        - focal_length_model (CNN): Focal length estimation model.
        - device (torch.device): Device (CPU or GPU) to use for computation.
        """
        self.file_path: str = file_path
        self.file_name, self.file_extension = split_file_name(self.file_path)
        self.media_type: str = get_media_type(self.file_extension)
        self.frames: List[FrameData] = []

        if self.media_type == "image":
            self.frame: np.ndarray = cv2.imread(self.file_path)
            self.total_frames: int = 1
            self.fps: int = 1
            self.height: int = self.frame.shape[0]
            self.width: int = self.frame.shape[1]
            self.color_channels: int = self.frame.shape[2]
            self.dpi: Tuple[int, int] = get_dpi(self.frame)
            self.pixel_size_mm: float = get_pixel_size_mm(self.height, self.width, self.dpi)

            focal_length_mm: float = estimate_focal_length(focal_length_model, self.frame, device)
            intrinsic_data: IntrinsicData = IntrinsicData(focal_length_mm, self.pixel_size_mm, self.height, self.width)
            extrinsic_data = ExtrinsicData(device)
            frame_data: FrameData = FrameData(self.frame, 0, intrinsic_data, extrinsic_data)
            self.frames.append(frame_data)

        if self.media_type == "video":
            video_cap: cv2.VideoCapture = read_video(self.file_path)
            self.total_frames: int = get_total_video_frames(video_cap)
            self.fps: int = get_video_fps(video_cap)
            self.height: int = None
            self.width: int = None
            self.color_channels: int = None
            self.dpi: Tuple[int, int] = None
            self.pixel_size_mm: float = None
            for i in tqdm(range(self.total_frames - 1), desc=f"Processing {self.file_name}{self.file_extension}"):
                frame: np.ndarray = get_video_frame(video_cap, i)
                if self.height is None:
                    self.height = frame.shape[0]
                if self.width is None:
                    self.width = frame.shape[1]
                if self.color_channels is None:
                    self.color_channels = frame.shape[2]
                if self.dpi is None:
                    self.dpi = get_dpi(frame)
                if self.pixel_size_mm is None:
                    self.pixel_size_mm = get_pixel_size_mm(self.height, self.width, self.dpi)

                focal_length_mm: float = estimate_focal_length(focal_length_model, frame, device)
                intrinsic_data: IntrinsicData = IntrinsicData(focal_length_mm, self.pixel_size_mm, self.height, self.width)
                extrinsic_data = ExtrinsicData(device)
                frame_data: FrameData = FrameData(frame, i, intrinsic_data, extrinsic_data)
                self.frames.append(frame_data)
            is_closed = close_video(video_cap)
            if not is_closed:
                raise ValueError(f"Video {self.file_path} failed to close")

        # Calculate mean intrinsic matrix based on all frames
        focal_lengths_mm = []
        focal_lengths_px = []
        for frame_data in self.frames:
            focal_lengths_mm.append(frame_data.intrinsic.focal_length_mm)
            focal_lengths_px.append(frame_data.intrinsic.focal_length_px)
        mean_focal_length_mm: float = np.mean(focal_lengths_mm)
        mean_focal_length_px: int = np.mean(focal_lengths_px)
        mean_intrinsic_matrix: np.ndarray = create_3x3_intrinsic_matrix(mean_focal_length_px, self.height, self.width)
        self.mean_intrinsic: IntrinsicData = IntrinsicData(mean_focal_length_mm, self.pixel_size_mm, self.height, self.width, mean_intrinsic_matrix)

    def interpolate_cam_pose(self):
        """
        Interpolate camera poses for frames in a video.

        This method interpolates camera rotation and translation matrices for frames where camera
        pose estimation was not successful.
        """
        if self.media_type == "image":
            raise ValueError(f"ERROR - Attempting to interpolate {self.media_type}. Interpolation only works with videos")
        
        last_solved_frame = None
        total_frames_interpolated = 0

        for i, frame in enumerate(self.frames):
            if frame.camera_solved:
                last_solved_frame = frame
            else:
                is_interpolating = True
                search_stride = 1

                while is_interpolating:
                    search_idx = i + search_stride
                    if search_idx >= self.total_frames - 1:
                        is_interpolating = False
                    else:
                        next_solved_frame = self.frames[search_idx]
                        if not next_solved_frame.camera_solved:
                            search_stride += 1
                            continue
                        else:
                            last_solved_frame_idx = last_solved_frame.frame_idx
                            next_solved_frame_idx = next_solved_frame.frame_idx
                            total_frame_gap = next_solved_frame_idx - last_solved_frame_idx
                            next_solved_frame_rotation = next_solved_frame.extrinsic.camera_rotation_matrix
                            next_solved_frame_translation = next_solved_frame.extrinsic.camera_translation
                            last_solved_frame_rotation = last_solved_frame.extrinsic.camera_rotation_matrix
                            last_solved_frame_translation = last_solved_frame.extrinsic.camera_translation
                            gap_rotation = next_solved_frame_rotation - last_solved_frame_rotation
                            gap_translation = next_solved_frame_translation - last_solved_frame_translation
                            rotation_per_frame = gap_rotation / total_frame_gap
                            translation_per_frame = gap_translation / total_frame_gap
                            stride = 1

                            for i in range(last_solved_frame_idx, next_solved_frame_idx):
                                r_mat = last_solved_frame_rotation + (rotation_per_frame * stride)
                                trans = last_solved_frame_translation + (translation_per_frame * stride)
                                self.frames[i].extrinsic.add_camera_rotation_matrix(r_mat)
                                self.frames[i].extrinsic.add_camera_translation(trans)
                                self.frames[i].set_cam_solved()
                                stride += 1
                                total_frames_interpolated += 1

        print(f"Successfully Interpolated Camera Motion for {total_frames_interpolated} Frames!")
        solved_frames = 0
        for frame in self.frames:
            if frame.camera_solved:
                solved_frames += 1
        print(f"Solved {solved_frames} of {self.total_frames} Frames")