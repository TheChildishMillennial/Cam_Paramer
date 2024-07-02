import os
import yaml
import gdown
from cam_paramer.utils.config import FOCAL_LENGTH_MODEL_PATH

def download_focal_length_checkpoint():
    """
    Downloads the focal length checkpoint file from Google Drive based on the model_configs.yaml file.

    Returns:
    bool: True if download is successful, False otherwise.
    """
    print("Attempting to download the focal length checkpoint...")
    
    # Determine the path to the configuration file
    config_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs', 'model_configs.yaml'))
    print(f"Config file path: {config_file_path}")

    # Check if the configuration file exists
    if not os.path.exists(config_file_path):
        print("Config file does not exist.")
        return False
    
    # Load model configurations from the YAML file
    with open(config_file_path, 'r') as config_file:
        model_configs = yaml.safe_load(config_file)

    focal_length = model_configs.get('focal_length', [])
    print(f"Model configs: {model_configs}")

    # Check if focal length model configuration is available
    if not focal_length:
        print("No focal length model config found.")
        return False

    # Extract file ID and download URL from model configurations
    file_id = focal_length[0].get('file_id') if len(focal_length) > 0 else None
    url = focal_length[1].get('url') if len(focal_length) > 1 else None
    print(f"File ID: {file_id}, URL: {url}")

    # Ensure file ID is available for downloading
    if not file_id:
        print("No file ID found for focal length model.")
        return False

    # Construct download URL and create directory for download
    download_url = f"https://drive.google.com/uc?id={file_id}"
    os.makedirs(os.path.dirname(FOCAL_LENGTH_MODEL_PATH), exist_ok=True)
    
    # Specify the correct path for download
    download_path = FOCAL_LENGTH_MODEL_PATH
    
    # Download the file using gdown
    gdown.download(download_url, download_path, quiet=False)

    # Check if download was successful
    if os.path.exists(download_path):
        # Check if the downloaded file is valid
        with open(download_path, 'rb') as f:
            if f.read(1) == b'<':
                print("Downloaded file is invalid (HTML content detected).")
                return False
    else:
        print(f"File was not found after download attempt: {download_path}")
        return False
    
    print(f"Focal Length Checkpoint downloaded successfully to: {download_path}")
    return True
