from torchvision import models
import torch.nn as nn

def build_vgg16(num_classes: int, freeze_until_feature_idx: int = -5):
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    # Odblokuj gradienty
    for p in model.features.parameters():
        p.requires_grad = True

    # Podmień ostatnią warstwę
    n_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(n_features, num_classes)

    # Zamroź wcześniejsze warstwy (transfer learning)
    if freeze_until_feature_idx is not None:
        for param in model.features[:freeze_until_feature_idx].parameters():
            param.requires_grad = False

    return model
