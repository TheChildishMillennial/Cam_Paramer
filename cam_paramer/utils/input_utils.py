import os
import yaml
from PIL import Image
import cv2
import math
import numpy as np
from typing import Tuple, List, Dict

TMP_FRAME_DIR = os.path.join(os.getcwd(), "temp_dir")

def split_file_name(file_path: str) -> Tuple[str, str]:
    """
    Split the file path into name and extension.

    Args:
    - file_path (str): Path to the file.

    Returns:
    Tuple[str, str]: File name and file extension in lowercase.
    """
    file_name, file_extension = os.path.splitext(os.path.basename(file_path))
    return file_name, file_extension.lower()

def get_supported_extensions() -> Tuple[List[str], List[str]]:
    """
    Get supported image and video extensions from configuration.

    Returns:
    Tuple[List[str], List[str]]: Tuple containing lists of image and video extensions.
    """
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    extension_data_path = os.path.join(parent_dir, "configs", "media_extension_configs.yaml")
    with open(extension_data_path, 'r') as extension_data_file:
        extension_data = yaml.safe_load(extension_data_file)
    image_extensions = extension_data['image_extensions']
    video_extensions = extension_data['video_extensions']
    return image_extensions, video_extensions

def get_media_type(file_extension: str) -> str:
    """
    Determine the type of media based on file extension.

    Args:
    - file_extension (str): File extension.

    Returns:
    str: "image" for image files, "video" for video files.

    Raises:
    ValueError: If the file extension is not supported.
    """
    image_extensions, video_extensions = get_supported_extensions()
    if file_extension in image_extensions:
        return "image"
    elif file_extension in video_extensions:
        return "video"
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")

def get_dpi(img: np.array) -> Tuple[int, int]:
    """
    Get the DPI (dots per inch) of an image.

    Args:
    - img (np.array): Input image as a NumPy array.

    Returns:
    Tuple[int, int]: Horizontal and vertical DPI of the image.
    """
    image = Image.fromarray(img)
    dpi = image.info.get('dpi', (72, 72))
    return dpi

def get_image_quality(img: np.array) -> float:
    """
    Calculate the image quality using Laplacian variance.

    Args:
    - img (np.array): Input image as a NumPy array.

    Returns:
    float: Image quality score.
    """
    quality = cv2.Laplacian(img, cv2.CV_64F).var()
    return quality

def get_pixel_size_mm(image_height: int, image_width: int, dpi: Tuple[int, int]) -> float:
    """
    Calculate the size of a pixel in millimeters based on image resolution and DPI.

    Args:
    - image_height (int): Height of the image in pixels.
    - image_width (int): Width of the image in pixels.
    - dpi (Tuple[int, int]): Horizontal and vertical DPI of the image.

    Returns:
    float: Size of a pixel in millimeters.
    """
    h_dpi, v_dpi = dpi    
    imp_h = image_height / v_dpi
    imp_w = image_width / h_dpi
    
    imp_diag_size = math.sqrt((imp_h **2) + (imp_w**2))
    res_diag_size = math.sqrt((image_height **2) + (image_width **2))

    pixel_size = ((imp_diag_size / res_diag_size) * 25.4) * 0.1
    return pixel_size

def search_input(input_path: str) -> List[Dict]:
    """
    Search for media files or directories containing media files in the input path.

    Args:
    - input_path (str): Input path which can be a file or directory.

    Returns:
    List[Dict]: List of dictionaries containing group information (group path, group name, media paths, media names).
    """
    groups = []

    # Handle single file input
    if not os.path.isdir(input_path):
        file_name, file_extension = split_file_name(input_path)
        group_dict = {
            'group_path': os.path.dirname(input_path),
            'group_name': file_name,
            'media_paths': [input_path],
            'media_names': [os.path.basename(input_path)]
        }
        groups.append(group_dict)
    
    # Handle directory input
    else:
        root_list = []
        dir_list = []
        file_list = []

        for root, dirs, files in os.walk(input_path):
            root_list.append(root)
            dir_list.append(dirs)
            file_list.append(files)

        # Process media files in the input directory
        input_dir_media = file_list[0]
        if len(input_dir_media) > 0:
            for group_idx, group_dir in enumerate(file_list[0]):
                group_dict = {
                    'group_path': root_list[0],
                    'group_name': os.path.splitext(group_dir)[0],
                    'media_paths': None,
                    'media_names': group_dir
                }
                f_paths = []
                for f in input_dir_media:
                    f_paths.append(os.path.join(root_list[0], f))
                group_dict['media_paths'] = f_paths
                groups.append(group_dict)
        
        # Process subdirectories within the input directory
        if len(dir_list[0]) > 0:
            for group_idx, group_dir in enumerate(dir_list[0]):
                group_dict = {
                    'group_path': root_list[group_idx + 1],
                    'group_name': group_dir,
                    'media_paths': None,
                    'media_names': file_list
                }
                f_paths = []
                for f in file_list[group_idx + 1]:
                    f_paths.append(os.path.join(root_list[group_idx + 1], f))
                group_dict['media_paths'] = f_paths
                groups.append(group_dict)

    return groups

def search_dir_for_media(dir_path: str) -> List[str]:
    """
    Search a directory for media files.

    Args:
    - dir_path (str): Path to the directory.

    Returns:
    List[str]: List of paths to the media files found in the directory.
    """
    media_paths = []
    for path in os.listdir(dir_path):
        media_path = os.path.join(dir_path, path)
        if not os.path.isdir(media_path):
            media_paths.append(media_path)
        else:
            print(f"WARNING! Not a media file. Skipping {media_path}")
    return media_paths
