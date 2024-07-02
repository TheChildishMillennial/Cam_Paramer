import os
import argparse
import sys

# Add the parent directory to the system path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from cam_paramer.media_queue.QueueData import QueueData
from cam_paramer.utils.system_utils import get_device
from cam_paramer.tracking.camera_solver import track_camera_motion
from cam_paramer.tracking.tracking_utils import visualize_camera_positions
from cam_paramer.utils.config import DEFAULT_IN_DIR, DEFAULT_OUT_DIR

def main(args):
    """
    Main function to solve camera motion and visualize camera positions for input media.

    Args:
    args (argparse.Namespace): Command-line arguments parsed by argparse.

    Raises:
    ValueError: If input or output directories are invalid.
    """
    input_path = args.input
    output_path = args.output
    interpolate_video_motion = args.interpolate_video_motion
    visualize_camera_motion = args.visualize_camera_motion
    device = get_device()

    # Initialize QueueData with input path
    try:
        queue = QueueData(input_path).media_groups
    except FileNotFoundError:
        raise ValueError(f"Input directory not found: {input_path}")

    for group_data in queue:
        print(f"Solving: {group_data.group_name}")
        
        # Track camera motion for each group
        try:
            track_camera_motion(group_data=group_data, device=device, frame_quality_threshold=80, interpolate_video_motion=interpolate_video_motion)
        except Exception as e:
            print(f"Error tracking camera motion for {group_data.group_name}: {str(e)}")
            continue

        solved_frames = 0
        unsolved_frames = 0

        # Iterate over media data in the group
        for media_data in group_data.media_group:
            for frame in media_data.frames:
                if frame.camera_solved:
                    solved_frames += 1
                    print(f"Frame {frame.frame_idx} Extrinsic: {frame.extrinsic.extrinsic_matrix}")
                else:
                    print(f"Could Not Solve: {frame.frame_number}")
                    unsolved_frames += 1

        print(f"Group: {group_data.group_name}\nSolved: {solved_frames} Frames\nUnsolved: {unsolved_frames} Frames")

        # Save extrinsic data to output directory
        try:
            group_data.save_all_extrinsic_data_json(output_path)
        except Exception as e:
            print(f"Error saving extrinsic data for {group_data.group_name}: {str(e)}")

        # Visualize camera positions
        if visualize_camera_motion:
            visualize_camera_positions(group_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=False, default=DEFAULT_IN_DIR, help="Path to input directory or media file")
    parser.add_argument("-o", "--output", required=False, default=DEFAULT_OUT_DIR, help="Path to output directory")
    parser.add_argument("-ivm", "--interpolate_video_motion", action='store_true', help="Interpolate Motion from Solved Frames?")
    parser.add_argument("-vis", "--visualize_camera_motion", action="store_true", help="Visualize Camera Locations After Solved")
    args = parser.parse_args()
    main(args)