import torch
import numpy as np
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import matplotlib.pyplot as plt


def plot_cam(model, image_paths, target_class=1):
    model.eval()
    device = next(model.parameters()).device

    # wykrycie warstwy CAM (jak w trainer)
    if hasattr(model, "features"):
        layer = [model.features[-1]]
    else:
        layer = [model.layer4[-1]]

    cam = GradCAM(model=model, target_layers=layer)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    fig, axs = plt.subplots(2, len(image_paths), figsize=(14, 6))

    for i, img_path in enumerate(image_paths):
        img = Image.open(img_path).convert("RGB")

        input_tensor = preprocess(img).unsqueeze(0).to(device)

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(target_class)]
        )[0]

        rgb_img = np.array(img.resize((224, 224))) / 255.0
        cam_vis = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        axs[0, i].imshow(cam_vis)
        axs[0, i].set_title("CAM")
        axs[0, i].axis("off")

        axs[1, i].imshow(rgb_img)
        axs[1, i].set_title("Original")
        axs[1, i].axis("off")

    plt.tight_layout()
    plt.show()
