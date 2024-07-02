import math
import cv2
from PIL import Image
from typing import Tuple
import numpy as np

def get_dpi(image: np.array) -> Tuple[int, int]:
    """
    Get the DPI (dots per inch) of an image.

    Args:
    - image (np.array): Input image as a NumPy array.

    Returns:
    Tuple[int, int]: Horizontal and vertical DPI of the image.
    """
    img = Image.fromarray(image)
    dpi = img.info.get('dpi', (72, 72))
    return dpi   

def get_pixel_size_mm(image: np.array) -> float:
    """
    Calculate the size of a pixel in millimeters based on image resolution.

    Args:
    - image (np.array): Input image as a NumPy array.

    Returns:
    float: Size of a pixel in millimeters.
    """
    height, width, _ = image.shape
    dpi = get_dpi(image)
    h_dpi, v_dpi = dpi    
    imp_h = height / v_dpi
    imp_w = width / h_dpi
    
    imp_diag_size = math.sqrt((imp_h **2) + (imp_w**2))
    res_diag_size = math.sqrt((height **2) + (width **2))

    pixel_size = ((imp_diag_size / res_diag_size) * 25.4) * 0.1
    return pixel_size


def calculate_blur(image_path: str) -> float:
    """
    Calculate the blur level of an image using Laplacian variance.

    Args:
    - image_path (str): Path to the input image file.

    Returns:
    float: Blur score of the image.
    """
    img = cv2.imread(image_path)
    blur = cv2.Laplacian(img, cv2.CV_64F).var()
    return blur
