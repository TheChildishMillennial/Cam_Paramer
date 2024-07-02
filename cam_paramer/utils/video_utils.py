import cv2
from typing import Tuple
import numpy as np

def read_video(video_path: str) -> cv2.VideoCapture:
    """
    Read a video file and return a VideoCapture object.

    Args:
    video_path (str): Path to the video file.

    Returns:
    cv2.VideoCapture: VideoCapture object for the video.
    
    Raises:
    ValueError: If the video file cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video {video_path}")
    
    return cap

def get_total_video_frames(video_cap: cv2.VideoCapture) -> int:
    """
    Get the total number of frames in a video.

    Args:
    video_cap (cv2.VideoCapture): VideoCapture object for the video.

    Returns:
    int: Total number of frames in the video.
    
    Raises:
    ValueError: If the video file is invalid or cannot be opened.
    """
    if not video_cap.isOpened():
        raise ValueError(f"Error: Could not open video")
    
    num_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if num_frames < 1:
        raise ValueError(f"Video File is invalid")
    
    return num_frames
    
def get_video_fps(video_cap: cv2.VideoCapture) -> int:
    """
    Get the frames per second (FPS) of a video.

    Args:
    video_cap (cv2.VideoCapture): VideoCapture object for the video.

    Returns:
    int: Frames per second of the video.
    
    Raises:
    ValueError: If the video file is invalid or cannot be opened.
    """
    if not video_cap.isOpened():
        raise ValueError(f"Error: Could not open video")
    
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    if fps < 1:
        raise ValueError(f"Video file is invalid")
    
    return int(fps)
    
def get_video_hw(video_cap: cv2.VideoCapture) -> Tuple[int, int]:
    """
    Get the height and width of the frames in a video.

    Args:
    video_cap (cv2.VideoCapture): VideoCapture object for the video.

    Returns:
    Tuple[int, int]: Height and width of the video frames.
    
    Raises:
    ValueError: If the video file is invalid or cannot be opened.
    """
    if not video_cap.isOpened():
        raise ValueError(f"Error: Could not open video")
    
    video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    return video_height, video_width

def get_video_frame(video_cap: cv2.VideoCapture, video_frame_index: int) -> np.ndarray:
    """
    Get a specific frame from a video.

    Args:
    video_cap (cv2.VideoCapture): VideoCapture object for the video.
    video_frame_index (int): Index of the frame to retrieve.

    Returns:
    np.ndarray: Frame at the specified index as a NumPy array.
    
    Raises:
    ValueError: If the video file is invalid, cannot be opened, or the frame cannot be read.
    """
    if not video_cap.isOpened():
        raise ValueError(f"Error: Could not open video")
    
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_index)
    ret, frame = video_cap.read()
    
    if not ret:
        raise ValueError(f"Error: Could not read frame at index {video_frame_index}")
    
    return frame

def close_video(video_cap: cv2.VideoCapture) -> bool:
    """
    Close a VideoCapture object.

    Args:
    video_cap (cv2.VideoCapture): VideoCapture object to close.

    Returns:
    bool: True if the video capture is successfully closed, False otherwise.
    """
    if not video_cap.isOpened():
        print(f"Video is already closed")
        return True
    
    video_cap.release()
    return not video_cap.isOpened()