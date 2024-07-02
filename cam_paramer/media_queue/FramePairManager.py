from cam_paramer.media_queue.FrameData import FrameData
from typing import List, Tuple

class FramePairManager:
    """
    Manager class for handling pairs of frames, categorizing them as solved or unsolved.
    """

    def __init__(self, input_frames: List[FrameData]) -> None:
        """
        Initialize FramePairManager with a list of input frames.

        Args:
        - input_frames (List[FrameData]): List of FrameData objects representing input frames.
        """
        self.unsolved_frames: List[FrameData] = input_frames
        self.solved_frames: List[FrameData] = []
        self.source_idx: int = 0
        self.target_idx: int = 0
        self.move_unsolved_frame_to_solved(0)

    def get_unsolved_frames(self) -> List[FrameData]:
        """
        Get the list of unsolved frames.

        Returns:
        - List[FrameData]: List of unsolved FrameData objects.
        """
        return self.unsolved_frames
    
    def get_solved_frames(self) -> List[FrameData]:
        """
        Get the list of solved frames.

        Returns:
        - List[FrameData]: List of solved FrameData objects.
        """
        return self.solved_frames

    def remove_item_from_unsolved(self, idx_to_remove: int) -> None:
        """
        Remove an item from the list of unsolved frames.

        Args:
        - idx_to_remove (int): Index of the frame to remove.
        """
        if self.unsolved_frames[idx_to_remove] is not None:
            self.unsolved_frames.pop(idx_to_remove)

    def remove_item_from_solved(self, idx_to_remove: int) -> None:
        """
        Remove an item from the list of solved frames.

        Args:
        - idx_to_remove (int): Index of the frame to remove.
        """
        if self.solved_frames[idx_to_remove] is not None:
            self.solved_frames.pop(idx_to_remove)
    
    def move_unsolved_frame_to_solved(self, unsolved_idx: int) -> None:
        """
        Move an unsolved frame to the list of solved frames.

        Args:
        - unsolved_idx (int): Index of the unsolved frame to move.
        """
        unsolved_frame = self.unsolved_frames[unsolved_idx]
        if unsolved_frame is not None:
            self.solved_frames.append(unsolved_frame)
            self.unsolved_frames.pop(unsolved_idx)
   
    def get_frame_pair(self) -> Tuple[FrameData, FrameData]:
        """
        Get a pair of frames consisting of one solved and one unsolved frame.

        Returns:
        - Tuple[FrameData, FrameData]: Pair of FrameData objects (solved frame, unsolved frame).
        
        Raises:
        - ValueError: If all frames are already solved.
        """
        solved_frames = list(reversed(self.solved_frames))
        if len(self.unsolved_frames) == 0:
            raise ValueError("ERROR - Trying to get frame pair but all frames are already solved")
        else:            
            source_frame: FrameData = solved_frames[self.source_idx]
            target_frame: FrameData = self.unsolved_frames[self.target_idx]
            return source_frame, target_frame
    
    def was_solved(self) -> bool:
        """
        Check if the current frame pair has been marked as solved.

        Returns:
        - bool: True if the frame pair was marked as solved, False otherwise.
        """
        is_finished: bool = False
        self.move_unsolved_frame_to_solved(self.target_idx)
        self.source_idx = 0
        self.target_idx = 0
        if len(self.unsolved_frames) == 0:
            is_finished = True
        return is_finished

    def was_unsolved(self) -> bool:
        """
        Check if the current frame pair has been marked as unsolved.

        Returns:
        - bool: True if the frame pair was marked as unsolved, False otherwise.
        """
        is_finished: bool = False
        self.source_idx += 1
        if self.source_idx >= len(self.solved_frames):
            self.source_idx = 0
            self.target_idx += 1
            if self.target_idx >= len(self.unsolved_frames):
                is_finished = True
        return is_finished
