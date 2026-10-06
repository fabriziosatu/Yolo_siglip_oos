"""
src/training/train.py
=======================
Entry point per il training della pipeline joint.

Uso:
    python src/training/train.py --siglip_variant vision_mlp --neg_ratio 0.5
    python src/training/train.py --siglip_variant completa --neg_ratio 0.25
    python src/training/train.py --resume weights/checkpoint_epoch010.pt
"""

import sys
import os
import argparse
from pathlib import Path

# Calcola la root del progetto dalla posizione DEL FILE (src/training/train.py
# -> risali due livelli), non dalla cartella di lancio. Prima usava
# sys.path.insert(0, ".") che funziona solo se lanci lo script dopo aver
# fatto "cd $HPC_ROOT" — se lo lanci da un'altra cartella (es. un test
# manuale), "import src..." fallisce con ModuleNotFoundError.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
# os.chdir esplicito: anche i path relativi USATI A RUNTIME (dataset_dir,
# detector.model_name, ecc. in config.py) sono relativi alla working
# directory, non solo gli import — senza questo, l'import funzionerebbe
# ma il caricamento dataset/pesi fallirebbe comunque se lanciato da
# una cartella diversa da $HPC_ROOT.
os.chdir(PROJECT_ROOT)

from src.training.trainer import Trainer
from src.utils.config     import CFG


def parse_args():
    parser = argparse.ArgumentParser(description="Training joint pipeline YOLO26 + SigLIPv2")
    parser.add_argument("--resume", type=str, default=None,
                         help="Path a un checkpoint da cui riprendere il training")
    parser.add_argument("--epochs", type=int, default=None,
                         help="Numero di epoche (sovrascrive config)")
    parser.add_argument("--batch", type=int, default=None,
                         help="Batch size (sovrascrive config)")
    parser.add_argument("--yolo_weights", type=str, default=None,
                         help="Path ai pesi YOLO (Fase 1 fine-tuned per opzione A, "
                              "checkpoint COCO base tipo yolo26n.pt per opzione B/scratch — "
                              "sovrascrive detector.model_name)")
    parser.add_argument("--skip_pretrained_load", action="store_true",
                         help="Opzione B — SigLIP parte da LoRA random-init sul backbone "
                              "pretrainato invece che dai pesi Fase 3 (il backbone SigLIP2 "
                              "resta comunque quello scaricato da HuggingFace, non e' random "
                              "puro — solo gli adapter LoRA lo sono, vedi discussione sul "
                              "significato di 'da zero')")
    parser.add_argument("--train_conf_threshold", type=float, default=None,
                         help="Soglia di confidenza per le ROI candidate in train mode "
                              "(sovrascrive detector.train_conf_threshold — default 0.2 "
                              "per pesi pretrained, valuta 0.1 se usi init da zero)")
    parser.add_argument("--roi_padding", type=float, default=None,
                         help="Padding attorno al box prima del RoI Align, stesso "
                              "significato di --padding in Fase 3 (sovrascrive "
                              "data.roi_padding, default 0.50 — coerente con i "
                              "checkpoint 'pad50' che carichi come pesi pretrained)")
    parser.add_argument("--max_rois_per_image", type=int, default=None,
                         help="Limite massimo di ROI per immagine passate a SigLIP "
                              "(sovrascrive detector.max_rois_per_image, default 20 — "
                              "riduce OOM/frammentazione CUDA da conteggio ROI variabile)")
    parser.add_argument("--raise_on_oom", action="store_true",
                         help="Modalita' diagnostica: un OOM fa fallire il job invece di "
                              "saltare il batch e continuare. Utile per test tipo 'fino a "
                              "che batch regge questa config?' — NON usarlo nei run che "
                              "vuoi portare a termine, perderesti l'intero job per un "
                              "singolo batch sfortunato.")
    parser.add_argument("--save_dir", type=str, default=None,
                         help="Cartella di output per checkpoint/history di questo run "
                              "(sovrascrive training.save_dir — usalo per separare gli "
                              "esperimenti, altrimenti si sovrascrivono a vicenda)")

    parser.add_argument("--siglip_variant", type=str, default=None,
                         choices=["vision_mlp", "completa"],
                         help="Quale architettura SigLIP usare")
    parser.add_argument("--neg_ratio", type=float, default=None,
                         help="Rapporto negativi sintetici/GT per immagine — guida SIA il "
                              "path del checkpoint Fase 3 da caricare SIA il mining di Fase 4 "
                              "(stesso significato di Fase 3 — 0.25 o 0.5)")
    parser.add_argument("--siglip_weights_dir", type=str, default=None,
                         help="Override manuale del path pesi Fase 3 (bypassa la derivazione "
                              "automatica da --neg_ratio)")
    parser.add_argument("--alpha", type=float, default=None,
                         help="Peso L_YOLO nella joint loss (sovrascrive config)")
    parser.add_argument("--beta", type=float, default=None,
                         help="Peso L_SigLIP nella joint loss (sovrascrive config)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.epochs:
        CFG.training.num_epochs = args.epochs
    if args.batch:
        CFG.data.batch_size = args.batch
    if args.yolo_weights:
        CFG.detector.model_name = args.yolo_weights
    if args.skip_pretrained_load:
        CFG.siglip.skip_pretrained_load = True
    if args.train_conf_threshold is not None:
        CFG.detector.train_conf_threshold = args.train_conf_threshold
    if args.roi_padding is not None:
        CFG.data.roi_padding = args.roi_padding
    if args.max_rois_per_image is not None:
        CFG.detector.max_rois_per_image = args.max_rois_per_image
    if args.raise_on_oom:
        CFG.training.raise_on_oom = True
    if args.save_dir:
        CFG.training.save_dir = Path(args.save_dir)
    if args.siglip_variant:
        CFG.siglip.variant = args.siglip_variant
    if args.neg_ratio is not None:
        CFG.negative_mining.neg_ratio = args.neg_ratio
    if args.siglip_weights_dir:
        CFG.siglip.pretrained_weights_dir = args.siglip_weights_dir
    if args.alpha is not None or args.beta is not None:
        if args.alpha is None or args.beta is None:
            raise ValueError("--alpha e --beta vanno passati insieme (alpha+beta deve fare 1)")
        CFG.training.alpha = args.alpha
        CFG.training.beta  = args.beta

    print(f"\n[Config] siglip.variant={CFG.siglip.variant}  neg_ratio={CFG.negative_mining.neg_ratio}")
    print(f"[Config] siglip.skip_pretrained_load={CFG.siglip.skip_pretrained_load} "
          f"({'OPZIONE B — SigLIP da zero' if CFG.siglip.skip_pretrained_load else 'OPZIONE A — SigLIP da Fase 3'})")
    print(f"[Config] alpha={CFG.training.alpha}  beta={CFG.training.beta}")
    print(f"[Config] detector.model_name={CFG.detector.model_name}")
    print(f"[Config] detector.train_conf_threshold={CFG.detector.train_conf_threshold}")
    print(f"[Config] data.roi_padding={CFG.data.roi_padding}")
    print(f"[Config] detector.max_rois_per_image={CFG.detector.max_rois_per_image}")
    print(f"[Config] training.raise_on_oom={CFG.training.raise_on_oom} "
          f"({'DIAGNOSTICO — il job fallisce su OOM' if CFG.training.raise_on_oom else 'paracadute attivo'})")
    print(f"[Config] training.save_dir={CFG.training.save_dir}\n")

    trainer = Trainer(cfg=CFG)

    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)

    history = trainer.train(start_epoch=start_epoch)