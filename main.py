import argparse
import os
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim

from src.utils.config import (
    Config,
    make_pneumonia_dataset_config,
    make_fracture_dataset_config,
)
from src.utils.misc import seed_everything, ensure_dirs
from src.data.pipeline import (
    build_datasets_and_loaders,
    plot_label_distribution,
    show_random_grid,
)
from src.models.vgg import build_vgg16
from src.models.resnet import build_resnet50
from src.training.trainer import Trainer
from src.utils.visualize import (
    plot_history,
    plot_confmat_and_report,
    plot_cam_evolution,
)

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--dataset_type",
        type=str,
        choices=["pneumonia", "fracture"],
        default="pneumonia",
    )

    p.add_argument(
        "--category",
        type=str,
        default="ELBOW",
        help="dla fracture: ELBOW, HAND, SHOULDER, FOREARM, FINGER",
    )

    p.add_argument(
        "--model",
        type=str,
        choices=["vgg16", "resnet"],
        default="vgg16",
    )

    p.add_argument("--epochs", type=int)
    p.add_argument("--batch_size", type=int)

    p.add_argument(
        "--use_original_split",
        action="store_true",
        help="dla pneumonia użyj oryginalnego podziału (bez chest_xray_new)",
    )

    # tryb visualize tylko z checkpointu
    p.add_argument(
        "--visualize",
        action="store_true",
        help="użyj zapisanego modelu i pokaż macierz pomyłek + raport (bez treningu)",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="nazwa checkpointu, np. epoch_9_acc_0.85.pth",
    )

    p.add_argument(
        "--no-cam",
        action="store_true",
        help="wyłącz rysowanie CAM",
    )

    return p.parse_args()



# =========================================================
# MAIN
# =========================================================

def main():
    args = parse_args()

    BASE_DIR = r"C:\Users\Weronika\Desktop\inzynierka\vgg16"

    # ----------------------------------
    # CONFIG injection – wybór datasetu
    # ----------------------------------
    if args.dataset_type == "pneumonia":
        dataset_cfg = make_pneumonia_dataset_config(BASE_DIR)
    else:
        dataset_cfg = make_fracture_dataset_config(BASE_DIR, args.category)

    cfg = Config(base_dir=BASE_DIR, dataset=dataset_cfg)

    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.epochs:
        cfg.dataset.max_epochs = args.epochs

    print(f"\nDataset: {cfg.dataset_type}")
    print("Path:", cfg.dataset_path)
    print("Klasy:", cfg.classes)
    print("Model:", args.model)
    print("--------------------------------------")

    # ----------------------------------
    # Seed + foldery
    # ----------------------------------
    seed_everything(cfg.seed)
    ensure_dirs(cfg.checkpoints_dir, cfg.samples_dir)

    # ----------------------------------
    # Pneumonia – ewentualny split 80/10/10
    # ----------------------------------
    if cfg.dataset_type == "pneumonia" and not args.use_original_split:
        chest_new = os.path.join(BASE_DIR, "archive", "chest_xray_new")
        if not os.path.exists(chest_new):
            print("Tworzę chest_xray_new...")
            from src.data.prepare_pneumonia_split import make_split
            make_split(cfg.dataset.path, chest_new)

        cfg.dataset.path = chest_new
        print("Używam chest_xray_new")

    # ----------------------------------
    # Load dataset
    # ----------------------------------
    image_datasets, dataloaders, sizes, class_names = build_datasets_and_loaders(
        cfg.dataset,
        cfg.batch_size,
        cfg.num_workers,
    )

    print("\nIlości danych:", sizes)

    # Podgląd obrazków + rozkład klas
    train_dir = os.path.join(cfg.dataset.path, "train")
    val_dir = os.path.join(cfg.dataset.path, "val")
    test_dir = os.path.join(cfg.dataset.path, "test")

    if os.path.exists(train_dir):
        show_random_grid(train_dir, class_names, n_per_class=4)

    plot_label_distribution(train_dir, val_dir, test_dir, class_names)

    # ----------------------------------
    # Class weights
    # ----------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.use_class_weights and "train" in image_datasets:
        labels = [lbl for _, lbl in image_datasets["train"]]
        counts = Counter(labels)
        total = sum(counts.values())
        class_weights = torch.tensor(
            [total / counts[i] for i in range(len(class_names))],
            dtype=torch.float32,
        ).to(device)
        print("⚖ Wagi klas:", class_weights.tolist())
    else:
        class_weights = None

    # ----------------------------------
    # Model
    # ----------------------------------
    freeze = cfg.dataset.freeze_until_feature_idx

    if args.model == "vgg16":
        model = build_vgg16(
            num_classes=len(class_names),
            freeze_until_feature_idx=freeze,
        )
    else:
        model = build_resnet50(
            num_classes=len(class_names),
            freeze_until_feature_idx=freeze,
        )

    model.to(device)

    # =========================================================
    # 🔥 POPRAWIONY BLOK CAM — JEDYNA ZMIANA
    # =========================================================
    cam_samples = []
    cam_class_id = cfg.dataset.cam_class_id

    if cfg.dataset.cam_enabled:
        test_dir = os.path.join(
            cfg.dataset.path,
            "test",
            cfg.dataset.cam_test_folder
        )

        if os.path.exists(test_dir):
            files = [
                os.path.join(test_dir, f)
                for f in os.listdir(test_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            cam_samples = files[: cfg.dataset.n_cam_samples]

    # =========================================================
    # TRYB VISUALIZE — BEZ TRENINGU
    # =========================================================
    if args.visualize:
        if not args.checkpoint:
            raise RuntimeError("Musisz podać checkpoint! --checkpoint NAZWA.pth")

        ckpt_path = os.path.join(cfg.checkpoints_dir, args.checkpoint)
        print(f"\n📦 Wczytuję checkpoint: {ckpt_path}")

        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        trainer = Trainer(
            device=device,
            model=model,
            criterion=nn.CrossEntropyLoss(weight=class_weights),
            optimizer=None,
            scheduler=None,
            train_dataloader=dataloaders.get("train"),
            val_dataloader=dataloaders.get("val"),
            test_dataloader=dataloaders.get("test"),
            checkpoint_path=cfg.checkpoints_dir,
            pneumonia_samples=cam_samples,
            cam_class_id=cam_class_id,
        )

        print("\n🧪 TESTING...\n")
        test_acc, test_loss, targets, preds, _ = trainer.test()

        plot_confmat_and_report(targets, preds, class_names)

        if cam_samples and not args.no_cam:
            plot_cam_evolution(
                cfg.samples_dir,
                cam_samples[: min(4, len(cam_samples))],
                cfg.max_epochs,
            )

        return

    # =========================================================
    # NORMALNY TRENING
    # =========================================================
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.step_size, gamma=cfg.gamma
    )

    trainer = Trainer(
        device=device,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=dataloaders.get("train"),
        val_dataloader=dataloaders.get("val"),
        test_dataloader=dataloaders.get("test"),
        checkpoint_path=cfg.checkpoints_dir,
        pneumonia_samples=cam_samples,
        cam_class_id=cam_class_id,
    )

    histories = []

    print("\nSTART TRAINING...\n")

    for epoch in range(1, cfg.max_epochs + 1):
        tr_acc, tr_loss = trainer.train(epoch, cfg.samples_dir)
        va_acc, va_loss = trainer.evaluate(epoch, cfg.samples_dir)

        histories.append({
            "train_acc": tr_acc,
            "train_loss": tr_loss,
            "val_acc": va_acc,
            "val_loss": va_loss,
        })

    print("\nTESTING BEST MODEL...\n")
    test_acc, test_loss, targets, preds, _ = trainer.test()

    plot_history(histories, test_acc=test_acc, test_loss=test_loss)
    plot_confmat_and_report(targets, preds, class_names)

    if cam_samples and not args.no_cam:
        plot_cam_evolution(
            cfg.samples_dir,
            cam_samples[: min(4, len(cam_samples))],
            cfg.max_epochs,
        )


if __name__ == "__main__":
    main()
