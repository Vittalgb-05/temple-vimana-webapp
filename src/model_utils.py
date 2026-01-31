import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import timm
import cv2
import numpy as np
import base64
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Mock classes if models.py not shared directly
from torchvision import models

class SegmentationModel(nn.Module):
    def __init__(self, encoder_name='resnet34', classes=1):
        super().__init__()
        try:
            self.model = smp.Unet(
                encoder_name=encoder_name, 
                encoder_weights=None, 
                in_channels=3, 
                classes=classes
            )
        except:
            print("SMP not available or failed. Using dummy.")
            self.model = nn.Identity()

    def forward(self, x):
        return self.model(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_models(seg_path, clf_path, num_classes=3):
    print(f"Loading models on {device}...")
    
    extra_msg = ""
    
    # Segmentation
    seg_model = SegmentationModel()
    if seg_path: # sampler check
        try:
            state = torch.load(seg_path, map_location=device)
            # Fix module. prefix
            new_state = {}
            if isinstance(state, dict):
                 # Check if state_dict is inside a key like 'state_dict' or 'model'
                keys_to_check = ['state_dict', 'model_state', 'model']
                for k in keys_to_check:
                    if k in state and isinstance(state[k], dict):
                        state = state[k]
                        break

            for k, v in state.items():
                if k.startswith('module.'):
                    new_state[k[7:]] = v
                else:
                    new_state[k] = v
            seg_model.load_state_dict(new_state, strict=False)
        except Exception as e:
            print(f"Failed to load seg checkpoint: {e}")
            extra_msg += f"Seg CP load error: {e}. "
    seg_model.to(device).eval()
    
    # Classifier (Match train_names.py structure)
    clf_model = models.densenet121(weights=None)
    # Correct the head immediately
    clf_model.classifier = nn.Linear(clf_model.classifier.in_features, num_classes)
    
    if clf_path:
        try:
            state = torch.load(clf_path, map_location=device)
            
            # Check for nested dict
            if isinstance(state, dict):
                keys_to_check = ['state_dict', 'model_state', 'model']
                for k in keys_to_check:
                    if k in state and isinstance(state[k], dict):
                        state = state[k]
                        break

            # Helper to find num_classes from weights
            if 'classifier.weight' in state:
                 w = state['classifier.weight']
                 print(f"Detected {w.shape[0]} classes from checkpoint.")
                 clf_model.classifier = nn.Linear(clf_model.classifier.in_features, w.shape[0])
            elif 'module.classifier.weight' in state:
                 w = state['module.classifier.weight']
                 print(f"Detected {w.shape[0]} classes from checkpoint.")
                 clf_model.classifier = nn.Linear(clf_model.classifier.in_features, w.shape[0])

            # Fix module. prefix
            new_state = {}
            for k, v in state.items():
                if k.startswith('module.'):
                    new_state[k[7:]] = v
                else:
                    new_state[k] = v
            
            clf_model.load_state_dict(new_state, strict=False)
        except Exception as e:
            print(f"Failed to load clf checkpoint: {e}")
            extra_msg += f"Clf CP load error: {e}. "
            
    clf_model.to(device).eval()
    
    return seg_model, clf_model, extra_msg

def encode_image(image_np):
    try:
        if image_np is None:
            print("ERROR: encode_image received None")
            return ""
        if image_np.dtype != np.uint8:
            print(f"WARNING: encode_image converting {image_np.dtype} to uint8")
            image_np = (image_np).astype(np.uint8)
            
        if len(image_np.shape) == 2:
            # Grayscale to BGR
            success, buffer = cv2.imencode('.jpg', cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR))
        else:
            # RGB to BGR
            success, buffer = cv2.imencode('.jpg', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            
        if not success:
            print("ERROR: cv2.imencode failed")
            return ""
        b64_str = base64.b64encode(buffer).decode('utf-8')
        print(f"Encoded image size: {len(b64_str)}")
        return b64_str
    except Exception as e:
        print(f"Exception in encode_image: {e}")
        return ""

def get_gradcam(model, input_tensor, rgb_image_float):
    # rgb_image_float: HxWx3, [0,1]
    # target_layers = [model.model.features[-1]] # DenseNet
    try:
        # Torchvision densenet
        target_layers = [model.features[-1]]
    except:
        # Fallback
        target_layers = [list(model.modules())[-1]]
    
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None) # Predicts highest class
    grayscale_cam = grayscale_cam[0, :]
    
    visualization = show_cam_on_image(rgb_image_float, grayscale_cam, use_rgb=True)
    return visualization
