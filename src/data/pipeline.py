import os
import random
from glob import glob
from typing import Dict, List

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ---------------------------------------------------------
# 1. Transformacje – rozdzielone dla pneumonia / fracture
# ---------------------------------------------------------
def get_transforms(dataset_type: str):
    if dataset_type == "fracture":
        # 🔥 IDENTYCZNE jak w notebooku!
        return {
            "train": transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ]),
            "val": transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ]),
            "test": transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ]),
        }

    # pneumonia → zostawiamy jak było
    return {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
        "val": transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
        "test": transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
    }


# ---------------------------------------------------------
# 2. Dataset + DataLoadery
# ---------------------------------------------------------
def build_datasets_and_loaders(
    dataset_root: str,
    batch_size: int,
    num_workers: int,
    dataset_type: str
):
    transforms_dict = get_transforms(dataset_type)

    splits = ["train", "val", "test"]
    existing = [s for s in splits if os.path.exists(os.path.join(dataset_root, s))]

    datasets_dict = {
        split: datasets.ImageFolder(
            os.path.join(dataset_root, split),
            transforms_dict[split]
        )
        for split in existing
    }

    # Notebook NIE stosował samplerów!
    dataloaders = {
        split: DataLoader(
            datasets_dict[split],
            batch_size=batch_size,
            shuffle=True if split == "train" else False,
            num_workers=num_workers
        )
        for split in existing
    }

    class_names = datasets_dict["train"].classes
    sizes = {s: len(datasets_dict[s]) for s in existing}

    return datasets_dict, dataloaders, sizes, class_names


# ---------------------------------------------------------
# 3. Wizualizacja klas
# ---------------------------------------------------------
def plot_label_distribution(train_path, val_path, test_path, classes):
    colors = sns.color_palette("Set2", len(classes))

    def count(split):
        if split is None or not os.path.exists(split):
            return [0] * len(classes)
        return [len(os.listdir(os.path.join(split, c))) for c in classes]

    train = count(train_path)
    val = count(val_path)
    test = count(test_path)
    total = [t + v + ts for t, v, ts in zip(train, val, test)]

    sets = ["Train", "Validation", "Test", "Total"]
    rows = [train, val, test, total]

    fig, axs = plt.subplots(1, 4, figsize=(14, 5))

    for i in range(4):
        bars = axs[i].bar(classes, rows[i], color=colors)
        axs[i].set_title(sets[i])
        total_count = sum(rows[i]) or 1
        for bar in bars:
            h = bar.get_height()
            axs[i].text(bar.get_x() + bar.get_width() / 2, h,
                        f"{int(h)} ({h/total_count:.1%})",
                        ha="center", va="bottom")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# 4. Pokaz losowych obrazów
# ---------------------------------------------------------
def show_random_grid(train_path: str, classes: List[str], n_per_class=4):
    fig, axs = plt.subplots(len(classes), n_per_class, figsize=(12, 4 * len(classes)))
    axs = np.array(axs).reshape(len(classes), n_per_class)

    for row, cls in enumerate(classes):
        files = os.listdir(os.path.join(train_path, cls))
        for col in range(n_per_class):
            img_path = os.path.join(train_path, cls, random.choice(files))
            img = Image.open(img_path).convert("RGB")

            axs[row][col].imshow(img)
            axs[row][col].set_title(cls)
            axs[row][col].axis("off")

    plt.tight_layout()
    plt.show()
