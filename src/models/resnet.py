import torch
import torch.nn as nn
from torchvision import models


def build_resnet50(num_classes: int = 2, freeze_features: bool = True):
    """
    Tworzy model ResNet50 kompatybilny z pipeline pneumonii i fracture.

    - wczytuje pretrenowany model
    - zamraża warstwy (tak jak VGG)
    - podmienia classifier (fc)
    """

    
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes),
    )

    return model
