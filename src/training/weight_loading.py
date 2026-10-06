"""
src/training/weight_loading.py
================================
Caricamento pesi Fase 3, versione corretta (fix doppio-wrapping LoRA).

PERCHE' QUESTE FUNZIONI SONO "GRANULARI" (una per componente) E VANNO
CHIAMATE DENTRO __init__, NON DOPO:
  get_peft_model() inietta i layer LoRA DENTRO il modulo base, in place.
  Se prima costruisci un adapter "a caso" (rank di config.py) con
  get_peft_model() e POI provi a caricare l'adapter vero (rank 16) sopra,
  il modulo base non e' piu' "pulito" — rischio di doppio-wrapping o
  mismatch silenzioso. Percio' queste funzioni vanno chiamate su un
  modulo base ANCORA VERGINE (self.base_vision_model prima di qualunque
  get_peft_model), dentro __init__ di SigLIPModule*, non come step
  separato dopo la costruzione.

Il rank/alpha dell'adapter NON servono piu' come parametri — PeftModel.
from_pretrained() li legge da solo da adapter_config.json dentro la
cartella salvata. Questo e' anche il motivo per cui "rank 16" non e' un
parametro che imposti in Fase 4: viaggia dentro il checkpoint stesso.

Nota su YOLO: nessuna funzione qui — YOLO26Detector carica gia' da solo
qualunque checkpoint passato in cfg.detector.model_name via Ultralytics.
"""

from pathlib import Path

import torch
from peft import PeftModel


class WeightLoadError(RuntimeError):
    pass


def _load_peft_adapter(base_module, adapter_dir: str):
    adapter_path = Path(adapter_dir)
    if not adapter_path.exists():
        raise WeightLoadError(f"Cartella adapter LoRA non trovata: '{adapter_dir}'.")
    try:
        peft_model = PeftModel.from_pretrained(base_module, str(adapter_path))
    except Exception as e:
        raise WeightLoadError(
            f"Impossibile caricare l'adapter LoRA da '{adapter_dir}': {e}\n"
            f"Verifica che la cartella contenga adapter_config.json e "
            f"adapter_model.safetensors/.bin (output di peft save_pretrained)."
        ) from e
    print(f"  ✓ Adapter LoRA caricato da {adapter_dir} "
          f"(rank/alpha letti automaticamente da adapter_config.json)")
    return peft_model


# ── Variante Vision+MLP (train_fase3_lora.py) ─────────────────────────────────

def load_siglip_vision_mlp_visual_only(base_vision_model, weights_dir: str):
    visual_dir = Path(weights_dir) / "lora_adapters"
    if not visual_dir.exists():
        raise WeightLoadError(
            f"Atteso 'lora_adapters' dentro '{weights_dir}' ma non trovato — "
            f"questo checkpoint non sembra della variante 'vision_mlp'. "
            f"Se hai un checkpoint 'completa', imposta siglip.variant='completa'."
        )
    return _load_peft_adapter(base_vision_model, str(visual_dir))


def load_siglip_vision_mlp_head_only(classifier_module, weights_dir: str):
    mlp_path = Path(weights_dir) / "mlp_head.pt"
    if not mlp_path.exists():
        raise WeightLoadError(f"Atteso 'mlp_head.pt' dentro '{weights_dir}' ma non trovato.")
    mlp_state = torch.load(mlp_path, map_location="cpu")
    missing, unexpected = classifier_module.load_state_dict(mlp_state, strict=False)
    if missing or unexpected:
        raise WeightLoadError(
            f"mlp_head.pt non compatibile con l'MLP (self.classifier) corrente.\n"
            f"  missing_keys:    {missing}\n"
            f"  unexpected_keys: {unexpected}\n"
            f"Probabile causa: mlp_hidden in config.py non combacia con "
            f"l'architettura usata per generare questo checkpoint, oppure il "
            f"nome dell'attributo in train_fase3_lora.py era diverso da "
            f"'classifier' — verificalo nello script."
        )
    print(f"  ✓ MLP classificatore caricato da {mlp_path}")


# ── Variante Completa (train_fase3_full_lora.py) ──────────────────────────────

def load_siglip_completa_visual_only(base_vision_model, weights_dir: str):
    visual_dir = Path(weights_dir) / "lora_adapters_visual"
    if not visual_dir.exists():
        raise WeightLoadError(
            f"Atteso 'lora_adapters_visual' dentro '{weights_dir}' ma non trovato — "
            f"questo checkpoint non sembra della variante 'completa'. "
            f"Se hai un checkpoint 'vision_mlp', imposta siglip.variant='vision_mlp'."
        )
    return _load_peft_adapter(base_vision_model, str(visual_dir))


def load_siglip_completa_text_only(base_text_model, weights_dir: str):
    text_dir = Path(weights_dir) / "lora_adapters_text"
    if not text_dir.exists():
        raise WeightLoadError(f"Atteso 'lora_adapters_text' dentro '{weights_dir}' ma non trovato.")
    return _load_peft_adapter(base_text_model, str(text_dir))


def load_siglip_completa_scale_bias(weights_dir: str):
    scale_bias_path = Path(weights_dir) / "scale_bias.pt"
    if not scale_bias_path.exists():
        raise WeightLoadError(f"Atteso 'scale_bias.pt' dentro '{weights_dir}' ma non trovato.")
    scale_bias = torch.load(scale_bias_path, map_location="cpu")
    print(f"  ✓ logit_scale/logit_bias caricati da {scale_bias_path}")
    return scale_bias["logit_scale"].float(), scale_bias["logit_bias"].float()
