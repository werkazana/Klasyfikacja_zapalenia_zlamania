import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms

def plot_history(histories, test_acc, test_loss):
    colors = sns.color_palette('Set1', 2)

    tr_acc = [h['train_acc'] for h in histories]
    va_acc = [h['val_acc'] for h in histories]
    tr_loss = [h['train_loss'] for h in histories]
    va_loss = [h['val_loss'] for h in histories]

    plt.rcParams['axes.grid'] = True
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(tr_acc, label='train', color=colors[0])
    plt.plot(va_acc, label='val', color=colors[1])
    plt.axhline(y=test_acc, linestyle='--', color='black',
                label=f'test: {test_acc:.4f}')
    plt.title('Accuracy')
    plt.legend()

    
    plt.subplot(1, 2, 2)
    plt.plot(tr_loss, label='train', color=colors[0])
    plt.plot(va_loss, label='val', color=colors[1])
    plt.axhline(y=test_loss, linestyle='--', color='black',
                label=f'test: {test_loss:.4f}')
    plt.title('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_confmat_and_report(targets, predictions, class_names):
    cm = confusion_matrix(targets, predictions)
    plt.rcParams['axes.grid'] = False

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    print(classification_report(targets, predictions,
                                target_names=class_names))

def plot_cam_evolution(samples_dir: str, image_paths, max_epochs: int):
    fig, axs = plt.subplots(len(image_paths),
                            max_epochs + 2,
                            figsize=(20, 2 * len(image_paths)))

    for row, img_path in enumerate(image_paths):
        orig = Image.open(img_path)

        # Original
        axs[row, 0].imshow(orig, cmap='gray')
        axs[row, 0].set_title('Original', fontsize=9)
        axs[row, 0].axis('off')

        base = os.path.splitext(os.path.basename(img_path))[0]

        # CAMs by epoch
        for col in range(1, max_epochs + 2):
            cam_path = os.path.join(
                samples_dir, f"{base}-cam-epoch-{col-1}.jpeg"
            )

            if os.path.exists(cam_path):
                cam_img = Image.open(cam_path)
                axs[row, col].imshow(cam_img, cmap='gray')
            else:
                axs[row, col].imshow([[0]], cmap='gray')

            axs[row, col].set_title(
                "Before FT" if col == 1 else f"Epoch {col-1}",
                fontsize=9
            )
            axs[row, col].axis("off")

    plt.tight_layout()
    plt.show()

def plot_cam_grid(model, image_paths, target_class=1):
    """
    Wyświetla CAM w formacie 2xN:
    - górny rząd: CAM
    - dolny rząd: oryginalny obraz
    """

    model.eval()
    device = next(model.parameters()).device

    # wybór warstwy CAM zależnie od modelu
    if hasattr(model, "features"):      # VGG16
        layer = [model.features[-1]]
    else:                               # ResNet50
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

        # CAM
        axs[0, i].imshow(cam_vis)
        axs[0, i].set_title("CAM")
        axs[0, i].axis("off")

        # Original
        axs[1, i].imshow(rgb_img)
        axs[1, i].set_title("Original")
        axs[1, i].axis("off")

    plt.tight_layout()
    plt.show()