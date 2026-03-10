import torch
import torch.nn as nn
from torchvision import models

class IncisionSeedModel(nn.Module):
    """
    EfficientNet-B1 adapted for keypoint regression (2 outputs: x, y).
    """
    def __init__(self, n_outputs=2, freeze_backbone=True):
        super().__init__()
        
        # Load EfficientNet-B1 without pre-defined weights to load local .pth later
        self.efficientnet = models.efficientnet_b1(weights=None)
        
        # The convolutional layers are in 'features'
        self.features = self.efficientnet.features 
        
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
        
        # Final fully connected layer (1280 features for B1)
        self.fc = nn.Linear(1280, n_outputs) 

    def forward(self, x):
        x = self.features(x) 
        x = x.mean([2, 3]) # Global Average Pooling
        x = self.fc(x)
        return x

    def unfreeze(self):
        for param in self.features.parameters():
            param.requires_grad = True
