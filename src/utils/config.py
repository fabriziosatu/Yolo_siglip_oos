"""
src/utils/config.py
====================
Configurazione pipeline joint YOLO26 + SigLIPv2 + MLP.

Rispetto alla versione precedente:
  - RIMOSSO: text encoder SigLIP, encode_prompts(), logits_pos/neg
  - AGGIUNTO: MLP classificatore binario (mlp_hidden, mlp_dropout)
  - INVARIATO: alpha/beta fissi, lr, lora_r, tutto il resto
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    data_dir:     Path = Path("data/processed_clean")
    dataset_yaml: Path = Path("data/processed_clean/dataset.yaml")
    img_size:     int  = 640
    roi_size:     int  = 224
    num_workers:  int  = 2
    batch_size:   int  = 4


@dataclass
class DetectorConfig:
    model_name:     str   = "weights/cluster_training/phase1_yolo_clean/weights/best.pt"
    conf_threshold: float = 0.25
    iou_threshold:  float = 0.45


@dataclass
class SigLIPConfig:
    model_name:        str   = "google/siglip2-base-patch16-224"
    lora_r_visual:     int   = 4
    lora_alpha_visual: int   = 8
    lora_dropout:      float = 0.30
    # MLP classificatore binario — sostituisce il text encoder
    mlp_hidden:        list  = field(default_factory=lambda: [256, 64])
    mlp_dropout:       float = 0.50


@dataclass
class TrainingConfig:
    num_epochs:  int   = 50
    lr_detector: float = 5e-6
    lr_siglip:   float = 5e-6

    weight_decay_yolo:   float = 1e-4
    weight_decay_siglip: float = 1e-2

    use_amp: bool = False

    # Pesi joint loss — fissi, calcolati con analyze_losses_custom.py
    # alpha = L_SigLIP / (L_YOLO + L_SigLIP) con vincolo alpha+beta=1
    alpha: float = 0.49
    beta:  float = 0.51

    save_dir:            Path = Path("weights/cluster_training/phase2_joint")
    save_every:          int  = 5
    early_stop_patience: int  = 20
    device:              str  = "cuda"


# Percorsi pesi — usati da evaluate.py e visualize_predictions.py
PHASE1_WEIGHTS = Path("weights/cluster_training/phase1_yolo_clean/weights/best.pt")
PHASE2_WEIGHTS = Path("weights/cluster_training/phase2_joint/best_model.pt")


@dataclass
class Config:
    data:     DataConfig     = field(default_factory=DataConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    siglip:   SigLIPConfig   = field(default_factory=SigLIPConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


CFG = Config()