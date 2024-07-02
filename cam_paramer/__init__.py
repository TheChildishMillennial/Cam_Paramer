import os
import sys

# Add the parent directory to the system path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from cam_paramer import tracking
from cam_paramer import focal_length
from cam_paramer import media_queue