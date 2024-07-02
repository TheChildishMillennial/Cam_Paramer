# Cam Paramer

Cam Paramer is a versatile tool designed for estimating both intrinsic and extrinsic camera parameters. The project employs advanced computer vision techniques such as DISK keypoint detection, LightGlue keypoint matching, and, as a fallback, SOLD2 line detection and matching. The software can process media in groups, separately, or as a single file.

## Key Features
- **Camera Parameter Estimation:** Estimates intrinsic and extrinsic camera parameters, including focal length, intrinsic matrix, extrinsic matrix, rotation, and translation vectors.
- **Advanced Keypoint Detection:** Utilizes DISK keypoint detection and LightGlue keypoint matching, with fallback to SOLD2 line detection.
- **Flexible Media Processing:** Process media in groups, separately, or as a single file. To make a group, simply add a directory to the input directory. For separate processing, place images and videos directly into the input directory. To process a single file, set the input path to the file.

## Installation
To set up the project, follow these steps:

1. **Clone the repository:**
    ```
    git clone https://github.com/your-repository/CamParamer.git
    cd CamParamer
    ```

2. **Create a virtual environment and activate it:**
    ```
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies:**
    ```
    pip install -r requirements.txt
    ```

4. **Optional: Install PyTorch with CUDA for GPU acceleration**
    If you want to run the project on a GPU with CUDA, you need to install the appropriate PyTorch version for your CUDA setup. Follow the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/) to install the correct version.

    Example for CUDA 11.8:
    ```
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    ```

## Usage
### Running Scripts
#### Camera Intrinsic Matrix Estimation
- Use `cam_intrinsic.py` to estimate the camera intrinsic matrix.
- Command:
    ```
    python cam_intrinsic.py [-i INPUT] [-o OUTPUT]
    ```
- **Automatic Mode:** Place files in `demo/input`, and results will be saved to `demo/output`.

#### Camera Extrinsic Data Estimation
- Use `cam_extrinsic.py` to estimate both the camera intrinsic and extrinsic matrices.
- Command:
    ```
    python cam_extrinsic.py [-i INPUT] [-o OUTPUT] [-ivm]
    ```
- **Automatic Mode:** Place files in `demo/input`, and results will be saved to `demo/output`.

### Command-Line Arguments
- `-i, --input`: Path to the input directory or file.
- `-o, --output`: Path to the output directory where results will be saved.
- `-ivm, --interpolate_video_motion`: Option to interpolate motion from solved frames (for `cam_extrinsic.py`).
- `-vis, --visualize_camera_motion`: Option to visualize the camera positions after motion is solved (for `cam_extrinsic.py`).

### Example
To estimate the camera intrinsic matrix for files in `demo/input` and save the results to `demo/output`:
```python cam_intrinsic.py -i demo/input -o demo/output
```

To estimate both the camera intrinsic and extrinsic matrices, with motion interpolation, for files in `demo/input` and save the results to `demo/output`:
```
python cam_extrinsic.py -i demo/input -o demo/output -ivm
```

### Note
- Ensure all necessary dependencies are installed as per the `requirements.txt`.
- For advanced usage and options like video motion interpolation, refer to additional command-line arguments in the scripts.

### Credit
This project incorporates components from the [MLFocalLengths](https://github.com/nandometzger/MLFocalLengths) repository.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.