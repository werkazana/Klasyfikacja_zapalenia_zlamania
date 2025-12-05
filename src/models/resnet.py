import torch.nn as nn
from torchvision import models

def build_resnet50(num_classes: int, freeze_until_feature_idx: int = None):
    """
    ResNet50 dostosowany do pipeline:
    - używa freeze_until_feature_idx (jak VGG)
    - ale NIE zamraża ostatniej warstwy konwolucyjnej (layer4),
      bo Grad-CAM potrzebuje gradientów z layer4[-1]
    """

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # warstwy konwolucyjne w kolejności używanej przez CAM
    layers = [
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4   # <-- OSTATNIA warstwa CAM! nie wolno zamrażać
    ]

    last_conv_idx = len(layers) - 1   # czyli 7

    # ---------------------------
    # ZAMRAŻANIE WARSTW
    # ---------------------------
    if freeze_until_feature_idx is not None:

        freeze_until = freeze_until_feature_idx

        # indeks ujemny → konwersja
        if freeze_until < 0:
            freeze_until = len(layers) + freeze_until

        # korekta, żeby CAM działał
        if freeze_until >= last_conv_idx:
            print("⚠️ ResNet freeze skorygowany: layer4 musi zostać odblokowany dla CAM!")
            freeze_until = last_conv_idx - 1

        print(f"🧊 ResNet50: zamrażam layers[:{freeze_until}]")

        for i, layer in enumerate(layers):
            if i < freeze_until:
                for param in layer.parameters():
                    param.requires_grad = False

    # ---------------------------
    # KOŃCOWY KLASYFIKATOR
    # ---------------------------
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes),
    )

    return model
