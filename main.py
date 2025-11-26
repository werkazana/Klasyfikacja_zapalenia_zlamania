import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from src.utils.config import Config
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

    p.add_argument("--dataset_type",
                   type=str,
                   default="fracture",
                   choices=["pneumonia", "fracture"])

    p.add_argument("--category",
                   type=str,
                   default="ELBOW")

    p.add_argument("--model",
                   type=str,
                   default="vgg16",
                   choices=["vgg16", "resnet"],
                   help="Wybór architektury: vgg16 lub resnet50")

    p.add_argument("--epochs", type=int)
    p.add_argument("--batch_size", type=int)

    p.add_argument("--visualize",
                   action="store_true",
                   help="Wizualizacja wyników bez treningu")

    p.add_argument("--checkpoint",
                   type=str,
                   default="best_model.pth")

    p.add_argument("--no-cam",
                   action="store_true",
                   help="Wyłącz CAM")
    p.add_argument(
        "--use_original_split",
                action="store_true",
                help="Użyj oryginalnego podziału chest_xray (bez chest_xray_new)",
    )

    return p.parse_args()


def main():
    args = parse_args()

    cfg = Config(
        dataset_type=args.dataset_type,
        current_category=args.category,
    )

    if args.epochs:
        cfg.max_epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size

    print(f"Dataset: {cfg.dataset_path}")
    print(f"Model: {args.model}")

    seed_everything(cfg.seed)
    ensure_dirs(cfg.checkpoints_dir, cfg.samples_dir)

    #PNEUMONIA
  
  
    if cfg.dataset_type == "pneumonia":

        print("\npodział pneumonia")
        dataset_path = cfg.dataset_path                      # oryginalny chest_xray
        new_dataset_path = os.path.join(cfg.base_dir, "archive", "chest_xray_new")

        if args.use_original_split:
            #ORYGINALNY chest_xray/train/val/test
            print("chest_xray (train/val/test)")
            dataset_base = dataset_path

        else:
            #podział (80/10/10)
            print("tworzę chest_xray_new (80/10/10)")

            if not os.path.exists(new_dataset_path):
                print("Tworzę chest_xray_new (80/10/10)")

                import shutil
                import random as pyrand

                for split in ["train", "val", "test"]:
                    for cls in ["NORMAL", "PNEUMONIA"]:
                        os.makedirs(os.path.join(new_dataset_path, split, cls), exist_ok=True)

                for cls in ["NORMAL", "PNEUMONIA"]:
                    all_files = []

                    for split in ["train", "val", "test"]:
                        src = os.path.join(dataset_path, split, cls)
                        files = os.listdir(src)
                        all_files.extend([(f, src) for f in files])

                    pyrand.shuffle(all_files)
                    n = len(all_files)

                    train_files = all_files[: int(n * 0.8)]
                    val_files   = all_files[int(n * 0.8):int(n * 0.9)]
                    test_files  = all_files[int(n * 0.9):]

                    for f, src in train_files:
                        shutil.copy(os.path.join(src, f),
                                    os.path.join(new_dataset_path, "train", cls, f))

                    for f, src in val_files:
                        shutil.copy(os.path.join(src, f),
                                    os.path.join(new_dataset_path, "val", cls, f))

                    for f, src in test_files:
                        shutil.copy(os.path.join(src, f),
                                    os.path.join(new_dataset_path, "test", cls, f))

                print("chest_xray_new utworzono\n")
            else:
                print("chest_xray_new już istnieje\n")

            dataset_base = new_dataset_path




    
    #FRACTURE
 
    else:
        dataset_base = cfg.dataset_path
        print("Korzystam z gotowego splitu fracture.\n")


    #DANE
    image_datasets, dataloaders, dataset_sizes, class_names = \
        build_datasets_and_loaders(dataset_base,
                                   cfg.batch_size,
                                   cfg.num_workers,
                                   cfg.dataset_type)

    print("\nClass names:", class_names)

    show_random_grid(os.path.join(dataset_base, "train"), class_names)
    plot_label_distribution(os.path.join(dataset_base, "train"),
                            os.path.join(dataset_base, "val"),
                            os.path.join(dataset_base, "test"),
                            class_names)

    if args.model == "vgg16":
        model = build_vgg16(num_classes=len(class_names),
                            freeze_until_feature_idx=cfg.freeze_until_feature_idx)

    elif args.model == "resnet":
        model = build_resnet50(num_classes=len(class_names))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.step_size, gamma=cfg.gamma
    )


    #CAM 
    pneumonia_samples = []
    if cfg.dataset_type == "pneumonia":
        pne_dir = os.path.join(dataset_base, "test", "PNEUMONIA")
        if os.path.exists(pne_dir):
            files = os.listdir(pne_dir)[:cfg.n_cam_samples]
            pneumonia_samples = [os.path.join(pne_dir, f) for f in files]

    trainer = Trainer(
        device,
        model,
        criterion,
        optimizer,
        scheduler,
        dataloaders["train"],       # train_dataloader
        dataloaders.get("val"),     # val_dataloader
        dataloaders.get("test"),    # test_dataloader
        cfg.checkpoints_dir,
        pneumonia_samples,
        cfg.cam_class_id
)


    if args.visualize:
        print("Wizualizacja\n")

        checkpoint_path = os.path.join(cfg.checkpoints_dir, args.checkpoint)
        ckpt = torch.load(checkpoint_path, map_location=device)

        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)

        model.eval()

        test_acc, test_loss, y_true, y_pred, _ = trainer.test()
        plot_confmat_and_report(y_true, y_pred, class_names)

        if pneumonia_samples and not args.no_cam:
            plot_cam_evolution(cfg.samples_dir, pneumonia_samples[:4], cfg.max_epochs)

        return

    #TRENING
   
    histories = []

    for ep in range(1, cfg.max_epochs + 1):
        print(f"\nEPOCH {ep}/{cfg.max_epochs}")

        tr_acc, tr_loss = trainer.train(ep, cfg.samples_dir)
        va_acc, va_loss = trainer.evaluate(ep, cfg.samples_dir)

        histories.append({
            "epoch": ep,
            "train_acc": tr_acc,
            "train_loss": tr_loss,
            "val_acc": va_acc,
            "val_loss": va_loss,
        })

 
    if dataloaders.get("test"):
        test_acc, test_loss, y_true, y_pred, _ = trainer.test()

        print(f"\nTest accuracy: {test_acc:.4f}")
        print(f"Test loss: {test_loss:.4f}")

        plot_history(histories, test_acc, test_loss)
        plot_confmat_and_report(y_true, y_pred, class_names)

        if pneumonia_samples and not args.no_cam:
            plot_cam_evolution(cfg.samples_dir, pneumonia_samples[:4], cfg.max_epochs)


if __name__ == "__main__":
    main()
