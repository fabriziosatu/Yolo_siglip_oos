"""
src/models/siglip_module.py
============================
Modulo SigLIPv2 con LoRA sul solo visual encoder + MLP classificatore binario.

Il text encoder e' stato rimosso su indicazione della professoressa.
Al suo posto, un MLP leggero classifica direttamente l'embedding visivo.

Architettura:
  roi_crops [N, 3, 224, 224]
      -> SigLIPv2 Visual Encoder + LoRA(r=8)
      -> pooler_output  [N, 768]
      -> L2 normalize
      -> MLP([768 -> 256 -> 64 -> 1])
      -> logit  [N, 1]   (usato con BCEWithLogitsLoss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from peft import LoraConfig, get_peft_model


EMBED_DIM = 768


class SigLIPModule(nn.Module):

    def __init__(
        self,
        model_name        = "google/siglip2-base-patch16-224",
        lora_r_visual     = 8,
        lora_alpha_visual = 16,
        lora_dropout      = 0.05,
        mlp_hidden        = None,
        mlp_dropout       = 0.30,
    ):
        super().__init__()

        if mlp_hidden is None:
            mlp_hidden = [256, 64]

        print(f"  Carico SigLIPv2 da '{model_name}'...")
        base_model = AutoModel.from_pretrained(model_name)

        # ── Visual encoder + LoRA ─────────────────────────────────────────────
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

        # gradient_checkpointing DISABILITATO — con 80GB VRAM non serve
        # e rallentava ogni batch di ~3x scambiando velocita' per memoria.
        # self.visual_encoder.gradient_checkpointing_enable()

        print(f"  Visual encoder — parametri trainabili:")
        self.visual_encoder.print_trainable_parameters()

        # ── MLP classificatore binario ────────────────────────────────────────
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

        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        arch_str = f"{EMBED_DIM} -> " + " -> ".join(str(h) for h in mlp_hidden) + " -> 1"
        print(f"  MLP classificatore: {arch_str}")

    def forward(self, roi_crops: torch.Tensor) -> torch.Tensor:
        """
        Args:
            roi_crops: (N, 3, 224, 224)
        Returns:
            logits: (N, 1)  >0 = vuoto, <0 = pieno
        """
        vis_out = self.visual_encoder(pixel_values=roi_crops)
        v = vis_out.pooler_output
        v = F.normalize(v, dim=-1)
        return self.classifier(v)

    def get_visual_embedding(self, roi_crops: torch.Tensor) -> torch.Tensor:
        vis_out = self.visual_encoder(pixel_values=roi_crops)
        return F.normalize(vis_out.pooler_output, dim=-1)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    print("Test SigLIPModule (MLP version, no gradient_checkpointing)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    siglip = SigLIPModule(
        lora_r_visual=8, lora_alpha_visual=16,
        lora_dropout=0.05, mlp_hidden=[256, 64], mlp_dropout=0.30,
    ).to(device)

    dummy = torch.rand(6, 3, 224, 224).to(device)
    logits = siglip(dummy)
    print(f"  logits: {logits.shape}  range [{logits.min().item():.3f}, {logits.max().item():.3f}]")

    loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    loss.backward()
    print(f"  loss: {loss.item():.4f}  ✓ Backward OK")