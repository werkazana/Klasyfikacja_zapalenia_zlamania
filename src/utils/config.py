from dataclasses import dataclass, field
from typing import Tuple, List
import os

@dataclass
class Config:
    # --- tryb pracy ---
    dataset_type: str = "pneumonia"  # "pneumonia" lub "fracture"
    current_category: str = "ELBOW"  # używane tylko przy fracture

    # --- ścieżki bazowe ---
    base_dir: str = r"C:\Users\Weronika\Desktop\inzynierka\vgg16"
    checkpoints_dir: str = os.path.join(base_dir, "checkpoints")
    samples_dir: str = os.path.join(base_dir, "samples")

    # --- ścieżki dla pneumonii ---
    pneumonia_path: str = os.path.join(base_dir, "archive", "chest_xray")
    pneumonia_classes: List[str] = field(default_factory=lambda: ["NORMAL", "PNEUMONIA"])

    # --- ścieżki dla złamań ---
    fracture_base_path: str = os.path.join(base_dir, "klasyfikacja_zlaman")
    available_categories: List[str] = field(default_factory=lambda: ["ELBOW", "HAND", "SHOULDER", "FOREARM", "FINGER"])

    # --- parametry uczenia ---
    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 16
    num_workers: int = 4
    max_epochs: int = 10
    lr: float = 5e-4
    step_size: int = 7
    gamma: float = 0.1
    freeze_until_feature_idx: int = -5
    seed: int = 27

    # --- Grad-CAM ---
    cam_class_id: int = 1
    n_cam_samples: int = 5

    # --- Balans danych (na razie tylko informacyjnie) ---
    use_sampler: bool = True
    use_class_weights: bool = True

    def __post_init__(self):
        """Automatyczne dostosowanie parametrów do rodzaju zbioru danych."""
        if self.dataset_type == "pneumonia":
            self.lr = 1e-4
            self.max_epochs = 10
            print("⚙️ Ustawiono parametry dla pneumonia: lr=5e-4, epochs=10")
        elif self.dataset_type == "fracture":
            self.lr = 1e-3
            self.max_epochs = 25
            print("⚙️ Ustawiono parametry dla fracture: lr=1e-3, epochs=25")

    @property
    def dataset_path(self) -> str:
        """Zwraca ścieżkę do aktywnego zbioru w zależności od typu."""
        if self.dataset_type == "pneumonia":
            return self.pneumonia_path
        elif self.dataset_type == "fracture":
            return os.path.join(
                self.fracture_base_path,
                f"XR_{self.current_category}_classification"
            )
        else:
            raise ValueError(f"Nieznany typ datasetu: {self.dataset_type}")

    @property
    def classes(self) -> List[str]:
        """Nazwy klas dla aktywnego typu danych."""
        if self.dataset_type == "pneumonia":
            return self.pneumonia_classes
        elif self.dataset_type == "fracture":
            return ["negative", "positive"]
