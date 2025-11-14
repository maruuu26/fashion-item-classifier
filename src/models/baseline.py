# src/models/baseline.py

import torch
import torch.nn as nn
from torchvision import models

def build_resnet18(num_classes=10, pretrained=True):
    """
    Load torchvision.models.resnet18(pretrained=pretrained),
    change the final FC layer to output `num_classes`,
    and return the model.
    - If you're on Apple Silicon: move to device "mps" when available, else "cpu".
    """
    # TODO: implement

    try:
        # New API: use weights enum
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    except Exception:
        # fallback for older API
        model = models.resnet18(pretrained=pretrained)
    
    # replace final classification layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# later in the training script add:
# device = "mps" if torch.backends.mps.is_available() else "cpu"
# model.to(device)
