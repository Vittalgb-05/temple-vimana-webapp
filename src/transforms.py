import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import torch

def get_preprocessing_transforms(size=256):
    return A.Compose([
        A.Resize(height=size, width=size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def prepare_image(image_bytes):
    """
    Reads image bytes, converts to RGB numpy array, original copy, and preprocessed tensor.
    """
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image

def preprocess_for_model(image, size=256):
    transform = get_preprocessing_transforms(size)
    augmented = transform(image=image)
    tensor = augmented['image'].unsqueeze(0) # (1, 3, H, W)
    return tensor

def crop_vimana(image, mask):
    """
    Crops the vimana from the image using the mask.
    image: HxWx3
    mask: HxW (0 or 1)
    """
    # Resize mask to overlap matches image
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
    # Find bounding box
    coords = cv2.findNonZero(mask)
    if coords is None:
        # Fallback: return center crop or whole image
        return image
        
    x, y, w, h = cv2.boundingRect(coords)
    
    # Optional padding
    pad = 10
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(image.shape[1], x + w + pad)
    y2 = min(image.shape[0], y + h + pad)
    
    return image[y1:y2, x1:x2]

def create_overlay(image, mask, color=(0, 255, 0), alpha=0.4):
    """
    Green overlay on vimana.
    """
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
    mask_bool = mask > 0
    overlay = image.copy()
    overlay[mask_bool] = color
    
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0).astype(np.uint8)
