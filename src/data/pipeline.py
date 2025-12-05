import os
import random
from typing import Dict, List, Tuple

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

from torchvision import datasets
from torch.utils.data import DataLoader


def build_datasets_and_loaders(
    dataset_cfg,
    batch_size: int,
    num_workers: int,
):
    """
    Buduje ImageFolder dla train/val/test na podstawie DatasetConfig.

    dataset_cfg.path      → katalog bazowy z podfolderami train/val/test
    dataset_cfg.transforms→ słownik transformacji: {"train": ..., "val": ..., "test": ...}
    """

    dataset_root = dataset_cfg.path          # np. XR_ELBOW_classification albo chest_xray_new
    transforms_dict = dataset_cfg.transforms # już wstrzyknięte z configa

    splits = ["train", "val", "test"]
    existing = [s for s in splits if os.path.exists(os.path.join(dataset_root, s))]

    datasets_dict = {
        split: datasets.ImageFolder(
            root=os.path.join(dataset_root, split),
            transform=transforms_dict[split]
        )
        for split in existing
    }

    dataloaders = {
        split: DataLoader(
            datasets_dict[split],
            batch_size=batch_size,
            shuffle=True if split == "train" else False,
            num_workers=num_workers,
        )
        for split in existing
    }

    class_names = datasets_dict["train"].classes
    sizes = {s: len(datasets_dict[s]) for s in existing}

    return datasets_dict, dataloaders, sizes, class_names


def plot_label_distribution(train_path: str, val_path: str, test_path: str, classes: List[str]):
    """
    Rysuje ile jest próbek w każdej klasie w zbiorach:
    - train
    - val
    - test
    - total
    """
    colors = sns.color_palette("Set2", len(classes))

    def count(split_path: str):
        if split_path is None or not os.path.exists(split_path):
            return [0] * len(classes)
        return [len(os.listdir(os.path.join(split_path, c))) for c in classes]

    train_counts = count(train_path)
    val_counts = count(val_path)
    test_counts = count(test_path)
    total = [t + v + ts for t, v, ts in zip(train_counts, val_counts, test_counts)]

    sets = ["Train", "Validation", "Test", "Total"]
    rows = [train_counts, val_counts, test_counts, total]

    plt.figure(figsize=(14, 5))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        bars = plt.bar(classes, rows[i], color=colors)
        plt.title(sets[i])
        total_count = sum(rows[i]) or 1
        for bar in bars:
            h = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                h,
                f"{int(h)} ({h/total_count:.1%})",
                ha="center",
                va="bottom",
                fontsize=8
            )
        plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


def show_random_grid(train_path: str, classes: List[str], n_per_class: int = 4):
    """
    Pokazuje siatkę losowych obrazków z każdej klasy z folderu train.
    """
    fig, axs = plt.subplots(len(classes), n_per_class,
                            figsize=(12, 4 * len(classes)))
    axs = np.array(axs).reshape(len(classes), n_per_class)

    for row, cls in enumerate(classes):
        cls_dir = os.path.join(train_path, cls)
        files = [f for f in os.listdir(cls_dir)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        if not files:
            continue

        for col in range(n_per_class):
            img_path = os.path.join(cls_dir, random.choice(files))
            img = Image.open(img_path).convert("RGB")

            axs[row][col].imshow(img, cmap="gray")
            axs[row][col].set_title(cls)
            axs[row][col].axis("off")

    plt.tight_layout()
    plt.show()
