Projekt obejmował klasyfikację zapalenia płuc oraz złamań kości różnych części ciała (łokieć, dłoń, ramię, przedramię, palec) z wykorzystaniem dwóch modeli głębokiego uczenia: VGG16 oraz ResNet50.
W systemie zastosowano transfer learning, fine-tuning, a także Class Activation Maps (CAM) do wizualizacji obszarów decyzyjnych modelu na obrazach RTG.

W projekcie wykorzystano dwa publiczne zbiory danych:

Chest X-Ray Pneumonia Dataset
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

(na podstawie pracy: https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5
)

XR Bones Dataset for Bone Fracture Detection
https://www.kaggle.com/datasets/japeralrashid/xr-bones-dataset-for-bone-fracture-detection

Projekt został zbudowany w architekturze modułowej z wykorzystaniem IOC (Inversion of Control), umożliwiając łatwe przełączanie między modelami i datasetami, a także generowanie CAM, wykresów treningu oraz macierzy pomyłek.
