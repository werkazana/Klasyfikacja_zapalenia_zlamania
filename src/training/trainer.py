import os
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt


class Trainer:
    def __init__(self,
                 device: torch.device,
                 model: torch.nn.Module,
                 criterion,
                 optimizer,
                 scheduler,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 test_dataloader: DataLoader,
                 checkpoint_path: str,
                 pneumonia_samples: List[str],
                 cam_class_id: int = 1):
        self.device = device
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.checkpoint_path = checkpoint_path
        self.best_accuracy = 0.0
        self.best_model_path = None
        self.pneumonia_samples = pneumonia_samples
        self.cam_class_id = cam_class_id
        os.makedirs(self.checkpoint_path, exist_ok=True)

        # --- automatyczne wykrywanie warstwy CAM ---
        self.cam_layer = self._detect_cam_layer()


    # ------------------------------------------------------------------------------
    # 🔍 Automatyczne wykrywanie warstwy CAM (VGG / ResNet)
    # ------------------------------------------------------------------------------
    def _detect_cam_layer(self):
        if hasattr(self.model, "features"):  
            # VGG16 → ostatnia warstwa konwolucyjna FEATURES
            return [self.model.features[-1]]
        elif hasattr(self.model, "layer4"):
            # ResNet50 → ostatnia warstwa konwolucyjna LAYER4
            return [self.model.layer4[-1]]
        else:
            raise ValueError("❌ Model nie ma ani features, ani layer4 — nie wiem, gdzie zrobić CAM.")


    def _cam_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])


    # ------------------------------------------------------------------------------
    # 🎨 Generowanie CAM
    # ------------------------------------------------------------------------------
    def save_cam_samples(self, epoch_nr: int, out_dir: str) -> None:
        if not self.pneumonia_samples:
            return

        self.model.eval()
        cam = GradCAM(model=self.model, target_layers=self.cam_layer)
        os.makedirs(out_dir, exist_ok=True)
        tfm = self._cam_transform()

        for image_path in self.pneumonia_samples:
            if not os.path.exists(image_path):
                continue

            img = Image.open(image_path).convert("RGB")
            x = tfm(img).unsqueeze(0).to(self.device)

            grayscale_cam = cam(
                input_tensor=x,
                targets=[ClassifierOutputTarget(self.cam_class_id)]
            )[0]

            rgb = np.array(img.resize((224, 224))) / 255.0
            vis = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)

            base = os.path.splitext(os.path.basename(image_path))[0]
            out = os.path.join(out_dir, f"{base}-cam-epoch-{epoch_nr}.jpeg")
            plt.imsave(out, vis)


    # ------------------------------------------------------------------------------
    # 🔥 TRENING
    # ------------------------------------------------------------------------------
    def train(self, current_epoch_nr: int, cam_out_dir: str) -> Tuple[float, float]:
        self.model.train()
        if current_epoch_nr == 1:
            self.save_cam_samples(0, cam_out_dir)

        run_loss, n_ok, total = 0.0, 0, 0
        loop = tqdm(self.train_dataloader, desc=f"Epoch {current_epoch_nr}")

        for x, y in loop:
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            y_hat = self.model(x)
            loss = self.criterion(y_hat, y)
            _, preds = torch.max(y_hat, 1)

            loss.backward()
            self.optimizer.step()

            run_loss += loss.item() * x.size(0)
            n_ok += torch.sum(preds == y).item()
            total += y.size(0)

            loop.set_postfix(
                train_acc=round(n_ok / total, 4),
                train_loss=round(run_loss / total, 4)
            )

        self.scheduler.step()
        return n_ok / total, run_loss / total

    # ------------------------------------------------------------------------------
    # 📊 WALIDACJA
    # ------------------------------------------------------------------------------
    def evaluate(self, current_epoch_nr: int, cam_out_dir: str) -> Tuple[float, float]:
        self.model.eval()
        run_loss, n_ok, total = 0.0, 0, 0

        with torch.no_grad():
            loop = tqdm(self.val_dataloader, desc=f"Validation {current_epoch_nr}")

            for x, y in loop:
                x, y = x.to(self.device), y.to(self.device)
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y)
                _, preds = torch.max(y_hat, 1)

                run_loss += loss.item() * x.size(0)
                n_ok += torch.sum(preds == y).item()
                total += y.size(0)

                loop.set_postfix(
                    val_acc=round(n_ok / total, 4),
                    val_loss=round(run_loss / total, 4)
                )

        val_acc, val_loss = n_ok / total, run_loss / total

        # CAM
        self.save_cam_samples(current_epoch_nr, cam_out_dir)

        # checkpoint
        if val_acc > self.best_accuracy:
            self.best_accuracy = val_acc
            ckpt = os.path.join(
                self.checkpoint_path,
                f"epoch_{current_epoch_nr}_acc_{val_acc:.2f}.pth"
            )
            torch.save(self.model.state_dict(), ckpt)
            self.best_model_path = ckpt
            print(f"💾 Zapisano najlepszy model: {ckpt}")

        return val_acc, val_loss


    # ------------------------------------------------------------------------------
    # 🧪 TEST
    # ------------------------------------------------------------------------------
    def test(self):
        if self.test_dataloader is None:
            print("⚠️ Brak test_dataloader — pomijam test.")
            return 0.0, 0.0, [], [], self.model

        self.model.eval()
        run_loss, n_ok, total = 0.0, 0, 0
        targets, preds_all = [], []

        with torch.no_grad():
            loop = tqdm(self.test_dataloader, desc="Testing")

            for x, y in loop:
                x, y = x.to(self.device), y.to(self.device)
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y)
                _, preds = torch.max(y_hat, 1)

                run_loss += loss.item() * x.size(0)
                n_ok += torch.sum(preds == y).item()
                total += y.size(0)

                targets.extend(y.cpu().numpy())
                preds_all.extend(preds.cpu().numpy())

                loop.set_postfix(
                    test_acc=round(n_ok / total, 4),
                    test_loss=round(run_loss / total, 4)
                )

        test_acc, test_loss = n_ok / total, run_loss / total

        if self.best_model_path:
            self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
            print(f"📦 Wczytano najlepszy model z: {self.best_model_path}")

        return test_acc, test_loss, targets, preds_all, self.model
