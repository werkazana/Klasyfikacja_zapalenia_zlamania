import os, random, shutil
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Ustawia ziarno losowości dla Pythona, NumPy i PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def ensure_dirs(*dirs: str) -> None:
    """Tworzy katalogi, jeśli nie istnieją."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def make_new_split_if_missing(dataset_path: str, new_dataset_path: str,
                              classes, split_fracs=(0.8, 0.1, 0.1)) -> None:
    """Tworzy nowy split train/val/test jeśli nie istnieje."""
    if os.path.exists(new_dataset_path):
        return

    train_frac, val_frac, test_frac = split_fracs

    for split in ["train", "val", "test"]:
        for cls in classes:
            os.makedirs(os.path.join(new_dataset_path, split, cls), exist_ok=True)

    for cls in classes:
        all_files = []
        for split in ["train", "val", "test"]:
            src = os.path.join(dataset_path, split, cls)
            if not os.path.exists(src):
                continue
            files = os.listdir(src)
            all_files.extend([(f, src) for f in files])

        random.shuffle(all_files)
        n = len(all_files)
        n_train = int(n * train_frac)
        n_val = int(n * (train_frac + val_frac))

        for f, src in all_files[:n_train]:
            shutil.copy(os.path.join(src, f), os.path.join(new_dataset_path, "train", cls, f))
        for f, src in all_files[n_train:n_val]:
            shutil.copy(os.path.join(src, f), os.path.join(new_dataset_path, "val", cls, f))
        for f, src in all_files[n_val:]:
            shutil.copy(os.path.join(src, f), os.path.join(new_dataset_path, "test", cls, f))
