import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import timm
import os

def create_dumb_models():
    os.makedirs('checkpoints', exist_ok=True)
    device = torch.device('cpu')
    
    print("Creating dummy Segmentation Model (UNet-ResNet34)...")
    try:
        seg_model = smp.Unet(encoder_name='resnet34', classes=1, encoder_weights=None)
    except:
        # Fallback if SMP not installed, matching app.py fallback logic would be good, 
        # but here we want to create a file that app.py *can* load.
        # If app.py uses SMP, we need a state dict matching SMP. 
        # If SMP is missing here, we can't create a valid state dict for it easily.
        # We assume requirements are satisfied.
        seg_model = nn.Conv2d(3, 1, 3, padding=1) # Mock
        
    torch.save(seg_model.state_dict(), 'checkpoints/seg_best.pth')
    print("Saved checkpoints/seg_best.pth")
    
    print("Creating dummy Classifier Model (DenseNet121)...")
    clf_model = timm.create_model('densenet121', pretrained=False, num_classes=3)
    torch.save(clf_model.state_dict(), 'checkpoints/clf_best.pth')
    print("Saved checkpoints/clf_best.pth")

if __name__ == "__main__":
    create_dumb_models()
