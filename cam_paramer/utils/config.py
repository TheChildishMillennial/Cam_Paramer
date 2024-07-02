import os

# Define paths for various configurations and directories
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FOCAL_LENGTH_MODEL_PATH = os.path.join(parent_dir, "cam_paramer", "focal_length", "checkpoints", "best_model.pth")
DEFAULT_IN_DIR = os.path.join(parent_dir, "demo", "input")
DEFAULT_OUT_DIR = os.path.join(parent_dir, "demo", "output")
DEFAULT_TMP_DIR = os.path.join(DEFAULT_IN_DIR, "tmp")
MODEL_CONFIGS_PATH = os.path.join(parent_dir, "configs", "model_configs.yaml")

# Optional: Add docstrings to clarify the purpose of each constant if necessary
"""
FOCAL_LENGTH_MODEL_PATH: Path to the focal length model checkpoint.
DEFAULT_IN_DIR: Default input directory path.
DEFAULT_OUT_DIR: Default output directory path.
DEFAULT_TMP_DIR: Temporary directory path within the input directory.
MODEL_CONFIGS_PATH: Path to the model configurations YAML file.
"""

# Additional comments or explanations can be added as needed
