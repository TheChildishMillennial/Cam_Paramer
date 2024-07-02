import os
import argparse
import sys

# Add the parent directory to the system path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from cam_paramer.media_queue.QueueData import QueueData
from cam_paramer.utils.config import DEFAULT_IN_DIR, DEFAULT_OUT_DIR

def main(args):
    """
    Main function to compute and save camera intrinsic matrices.

    Args:
    args (argparse.Namespace): Command-line arguments parsed by argparse.
    """
    input_path = args.input
    output_path = args.output

    try:
        queue = QueueData(input_path).media_groups
    except FileNotFoundError:
        raise ValueError(f"Input directory not found: {input_path}")

    print(f"Processing {len(queue)} groups from input: {input_path}")
    
    for group_data in queue:
        print(f"Group: {group_data.group_name}")
        
        for media_data in group_data.media_group:
            print(f"Intrinsic 3x3: {media_data.mean_intrinsic.intrinsic_matrix}")
            print(f"Focal Length mm: {media_data.mean_intrinsic.focal_length_mm}")
            print(f"Focal Length px: {media_data.mean_intrinsic.focal_length_px}")

            try:
                media_data.mean_intrinsic.save_intrinsic_matrix_3x3_json(f"intrinsic_matrix_{group_data.group_name}", output_path)
                print(f"Successfully Saved {group_data.group_name}.json to {output_path}")
            except Exception as e:
                print(f"Error saving intrinsic matrix for {group_data.group_name}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=False, default=DEFAULT_IN_DIR, help="Path to input directory or media file")
    parser.add_argument("-o", "--output", required=False, default=DEFAULT_OUT_DIR, help="Path to output directory")
    args = parser.parse_args()
    main(args)