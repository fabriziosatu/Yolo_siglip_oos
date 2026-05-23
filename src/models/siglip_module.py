"""
src/models/siglip_module.py
============================
Modulo SigLIPv2 con LoRA sul solo visual encoder + MLP classificatore binario.

Il text encoder è stato rimosso su indicazione della professoressa.
Al suo posto, un MLP leggero classifica direttamente l'embedding visivo.

Architettura:
  roi_crops [N, 3, 224, 224]
      → SigLIPv2 Visual Encoder + LoRA(r=8)   [invariato rispetto all'originale]
      → pooler_output  [N, 768]
      → L2 normalize
      → MLP([768 → 256 → 64 → 1])
      → logit  [N, 1]   (usato con BCEWithLogitsLoss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from peft import LoraConfig, get_peft_model


# Dimensione embedding SigLIPv2-base
EMBED_DIM = 768


class SigLIPModule(nn.Module):
    """
    SigLIPv2 visual encoder (frozen + LoRA) + MLP classificatore binario.

    Rispetto alla versione originale:
      - RIMOSSO: text encoder, tokenizer, encode_prompts(), logit_scale/bias,
                 text_embeds_pos/neg, lora_r_text, lora_alpha_text
      - AGGIUNTO: MLP binario (768 → 256 → 64 → 1)
      - INVARIATO: visual encoder, LoRA su q_proj/v_proj, gradient_checkpointing
    """

    def __init__(
        self,
        model_name:        str   = "google/siglip2-base-patch16-224",
        lora_r_visual:     int   = 8,
        lora_alpha_visual: int   = 16,
        lora_dropout:      float = 0.05,
        mlp_hidden:        list  = None,
        mlp_dropout:       float = 0.30,
    ):
        super().__init__()

        if mlp_hidden is None:
            mlp_hidden = [256, 64]

        print(f"  Carico SigLIPv2 da '{model_name}'...")
        base_model = AutoModel.from_pretrained(model_name)

        # ── Visual encoder + LoRA ─────────────────────────────────────────────
        # Identico all'originale — stessi target_modules, stesso schema LoRA
        lora_cfg_visual = LoraConfig(
            r              = lora_r_visual,
            lora_alpha     = lora_alpha_visual,
            target_modules = ["q_proj", "v_proj"],
            lora_dropout   = lora_dropout,
            bias           = "none",
        )
        self.visual_encoder = get_peft_model(
            base_model.vision_model, lora_cfg_visual
        )
        self.visual_encoder.gradient_checkpointing_enable()
        print(f"  Visual encoder — parametri trainabili:")
        self.visual_encoder.print_trainable_parameters()

        # ── MLP classificatore binario ────────────────────────────────────────
        # Sostituisce il text encoder + similarità coseno.
        # Input: embedding visivo L2-normalizzato (768-dim)
        # Output: logit singolo → sigmoid → P(scaffale vuoto)
        layers = []
        in_dim = EMBED_DIM
        for h in mlp_hidden:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(mlp_dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))

        self.classifier = nn.Sequential(*layers)

        # Inizializzazione conservativa per evitare saturazione iniziale
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        arch_str = f"{EMBED_DIM} → " + " → ".join(str(h) for h in mlp_hidden) + " → 1"
        print(f"  MLP classificatore: {arch_str}")

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, roi_crops: torch.Tensor) -> torch.Tensor:
        """
        Calcola il logit binario per ogni ROI crop.

        Args:
            roi_crops: (N, 3, 224, 224) — crop RGB estratti dalle bounding box

        Returns:
            logits: (N, 1)
                    > 0  → scaffale probabilmente vuoto
                    < 0  → scaffale probabilmente pieno
        """
        vis_out = self.visual_encoder(pixel_values=roi_crops)
        v = vis_out.pooler_output          # (N, 768)
        v = F.normalize(v, dim=-1)         # normalizzazione L2
        return self.classifier(v)          # (N, 1)

    def get_visual_embedding(self, roi_crops: torch.Tensor) -> torch.Tensor:
        """
        Utility: restituisce solo l'embedding visivo normalizzato.
        Utile per visualizzazioni e analisi.

        Args:
            roi_crops: (N, 3, 224, 224)
        Returns:
            embeddings: (N, 768)
        """
        vis_out = self.visual_encoder(pixel_values=roi_crops)
        return F.normalize(vis_out.pooler_output, dim=-1)


# ── Test rapido ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    print("Test SigLIPModule (MLP version)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    siglip = SigLIPModule(
        model_name        = "google/siglip2-base-patch16-224",
        lora_r_visual     = 8,
        lora_alpha_visual = 16,
        lora_dropout      = 0.05,
        mlp_hidden        = [256, 64],
        mlp_dropout       = 0.30,
    ).to(device)

    dummy_crops = torch.rand(6, 3, 224, 224).to(device)
    logits = siglip(dummy_crops)

    print(f"\n  logits shape : {logits.shape}")   # atteso (6, 1)
    print(f"  logits range : [{logits.min().item():.3f}, {logits.max().item():.3f}]")

    loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    loss.backward()
    print(f"  BCE loss     : {loss.item():.4f}")
    print("  ✓ Backward OK")

    embeds = siglip.get_visual_embedding(dummy_crops)
    print(f"  embeddings   : {embeds.shape}")   # atteso (6, 768)

    print("\n✓ SigLIPModule (MLP) OK!")