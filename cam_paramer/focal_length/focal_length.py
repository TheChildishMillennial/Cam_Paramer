import os
import argparse
import torch
import torch.nn as nn
import cv2
from torchvision import transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
import kornia
import numpy as np
import json
from cam_paramer.utils.config import FOCAL_LENGTH_MODEL_PATH, DEFAULT_IN_DIR, DEFAULT_OUT_DIR
from cam_paramer.utils.checkpoint_utils import download_focal_length_checkpoint

device = kornia.utils.get_cuda_or_mps_device_if_available()

class CNN(nn.Module):
    """CNN model for focal length estimation."""
    def __init__(self):
        super(CNN, self).__init__()
        self.model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        self.fc1 = nn.Linear(1000, 1)
        self.lossl1 = torch.nn.L1Loss()
        self.lossl2 = torch.nn.MSELoss()
        self.log_transform = True
        self.soft_plus = nn.Softplus()

    def forward(self, x):
        x = self.model(x)
        x = self.fc1(x)
        return x

    def get_loss(self, pred, gt, eps=1e-7): 
        """Calculate optimization loss."""
        mean_pred = pred.mean()
        mean_gt = gt.to(float).mean()

        lossl1 = self.lossl1(pred, gt)
        lossl2 = self.lossl2(pred, gt)

        if self.log_transform:
            pred = self.soft_plus(pred)
            pred_log = torch.log(pred + eps)
            gt_log = torch.log(gt + eps)
            lossl1_log = self.lossl1(pred_log, gt_log)
            lossl2_log = self.lossl2(pred_log, gt_log)
            optimization_loss = lossl1_log
        else:
            optimization_loss = lossl1

        return optimization_loss, {
            "lossl1": lossl1.detach().item(), "lossl2": lossl2.detach().item(), 
            "lossl1_log": lossl1_log.detach().item(), "lossl2_log": lossl2_log.detach().item(), 
            "mean_pred": mean_pred.detach().item(), "mean_gt": mean_gt.detach().item(), 
            "optimization_loss": optimization_loss.detach().item()
        }

def load_focal_length_model(model_path: str, device: torch.device) -> CNN:
    """Load pretrained focal length estimation model."""
    print(f"Loading focal length model from {model_path}...")
    if not os.path.exists(model_path):
        success = download_focal_length_checkpoint()
        if not success:
            raise RuntimeError("ERROR! - Focal Length Model Failed to Download")
    model = CNN()
    checkpoint = torch.load(model_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    torch.set_grad_enabled(False)
    checkpoint_name = os.path.basename(model_path)
    print(f"Focal Length Estimation Checkpoint: {checkpoint_name}\nLoaded on Device: {device}")
    return model

def preprocess_image(image: np.array) -> torch.Tensor:    
    """Preprocess image for model input."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    processed_image = transform(image)
    processed_image = processed_image.unsqueeze(0)
    return processed_image

def estimate_focal_length(model: CNN, image: np.array, device: torch.device) -> float:
    """Estimate focal length from input image."""
    image_tensor = preprocess_image(image)    

    try:
        image_tensor = image_tensor.to(device)
        focal_length = model(image_tensor)
    except RuntimeError as e:
        if 'out of memory' in str(e):
            torch.cuda.empty_cache()
            print("CUDA out of memory. Switching to CPU...")
            device = torch.device("cpu")
            model.to(device)
            image_tensor = image_tensor.to(device)
        else:
            raise e   
    
    focal_length_value = focal_length.item()
    return focal_length_value        

def main(args):
    """Main function to run focal length estimation."""
    image = cv2.imread(args.input_image)
    focal_length_model = load_focal_length_model(FOCAL_LENGTH_MODEL_PATH, args.device)
    est_focal_length = estimate_focal_length(focal_length_model, image, args.device)
    print(f"Estimated Focal Length: {est_focal_length}")
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    file_name = os.path.split(os.path.basename(args.input_image))[1]
    file_path = os.path.join(args.output_path, f"focal_length_{file_name}.json")
    focal_length_data = {"focal_length": est_focal_length}
    with open(file_path, 'w') as f:
        json.dump(focal_length_data, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_image", required=False, default=DEFAULT_IN_DIR, type=str, help="Path to input image")
    parser.add_argument("-o", "--output_path", required=False, default=DEFAULT_OUT_DIR, type=str, help="Path to output directory")
    parser.add_argument("-d", "--device", required=False, default=device, type=str, help="Device to load model on (e.g., 'cpu', 'cuda')")
    args = parser.parse_args()
    main(args)
