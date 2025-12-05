from dataclasses import dataclass
from typing import Dict, List, Tuple
import os
from torchvision import transforms


# ============================================================
#  Konfiguracja pojedynczego datasetu (IOC)
# ============================================================

@dataclass
class DatasetConfig:
    name: str
    path: str
    classes: List[str]
    transforms: Dict[str, transforms.Compose]

    lr: float
    max_epochs: int
    step_size: int
    gamma: float
    freeze_until_feature_idx: int

    use_sampler: bool
    use_class_weights: bool

    # CAM (parametry)
    cam_enabled: bool = False
    cam_class_id: int | None = None
    cam_test_folder: str | None = None
    n_cam_samples: int = 0

    image_size: Tuple[int, int] = (224, 224)


# ============================================================
#  Główny CONFIG – trzyma DatasetConfig (pełne IOC)
# ============================================================

@dataclass
class Config:
    base_dir: str
    dataset: DatasetConfig

    batch_size: int = 16
    num_workers: int = 4
    seed: int = 27

    def __post_init__(self):
        self.checkpoints_dir = os.path.join(self.base_dir, "checkpoints")
        self.samples_dir = os.path.join(self.base_dir, "samples")

    # shortcuty
    @property
    def dataset_type(self):
        return self.dataset.name

    @property
    def dataset_path(self):
        return self.dataset.path

    @property
    def classes(self):
        return self.dataset.classes

    @property
    def image_size(self):
        return self.dataset.image_size

    @property
    def lr(self):
        return self.dataset.lr

    @property
    def max_epochs(self):
        return self.dataset.max_epochs

    @property
    def step_size(self):
        return self.dataset.step_size

    @property
    def gamma(self):
        return self.dataset.gamma

    @property
    def freeze_until_feature_idx(self):
        return self.dataset.freeze_until_feature_idx

    @property
    def use_sampler(self):
        return self.dataset.use_sampler

    @property
    def use_class_weights(self):
        return self.dataset.use_class_weights

    @property
    def cam_enabled(self):
        return self.dataset.cam_enabled

    @property
    def cam_test_folder(self):
        return self.dataset.cam_test_folder

    @property
    def cam_class_id(self):
        return self.dataset.cam_class_id

    @property
    def n_cam_samples(self):
        return self.dataset.n_cam_samples


# ============================================================
#  Fabryki IOC – poprawne
# ============================================================

def make_pneumonia_dataset_config(base_dir: str) -> DatasetConfig:
    path = os.path.join(base_dir, "archive", "chest_xray")
    image_size = (224, 224)

    transforms_dict = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
            transforms.RandomRotation(15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(image_size[0]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size[0]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size[0]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
    }

    return DatasetConfig(
        name="pneumonia",
        path=path,
        classes=["NORMAL", "PNEUMONIA"],
        transforms=transforms_dict,
        lr=1e-4,
        max_epochs=10,
        step_size=7,
        gamma=0.1,
        freeze_until_feature_idx=-5,
        use_sampler=False,
        use_class_weights=False,

        cam_enabled=True,
        cam_class_id=1,
        cam_test_folder="PNEUMONIA",
        n_cam_samples=5,

        image_size=image_size,
    )


def make_fracture_dataset_config(base_dir: str, category: str) -> DatasetConfig:
    path = os.path.join(base_dir, "klasyfikacja_zlaman", f"XR_{category}_classification")
    image_size = (224, 224)

    transforms_dict = {
        "train": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.CenterCrop(image_size[0]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
        "val": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(image_size[0]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
        "test": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(image_size[0]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
    }

    return DatasetConfig(
        name="fracture",
        path=path,
        classes=["negative", "positive"],
        transforms=transforms_dict,
        lr=1e-4,
        max_epochs=10,
        step_size=5,
        gamma=0.1,
        freeze_until_feature_idx=20,
        use_sampler=False,
        use_class_weights=True,
        cam_enabled=True,
        cam_class_id=1,
        cam_test_folder="positive",
        n_cam_samples=5,

        image_size=image_size,
    )
