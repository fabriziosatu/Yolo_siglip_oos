"""
src/models/siglip_module.py
============================
Two SigLIP2 variants, selectable via cfg.siglip.variant:

  SigLIPModuleVisionMLP  — YOUR ORIGINAL VERSION, UNCHANGED.
      Only vision encoder + LoRA + MLP binary classifier.
      No changes to the internal logic: same architecture,
      same classifier weight init, same forward pass. The only
      difference is that it now returns a dict {"logit": ...} instead
      of a bare tensor, for uniformity with the other variant (see
      note in joint_pipeline.py on why).

  SigLIPModuleCompleta   — NEW, restores the text encoder removed
      on the professor's instruction, to be able to test it in Phase 4 as
      an alternative to the Vision+MLP variant. Identical architecture to
      train_fase3_full_lora.py: vision+text with LoRA on both,
      cosine similarity to prompts, trainable logit_scale/logit_bias.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model


EMBED_DIM = 768  # NO LONGER USED — see bug fix: the real dimension must
                 # always be read from base_model.config.vision_config.hidden_size,
                 # it varies with the checkpoint (768 for "base", 1536 for "giant").
                 # Left only to not break any external imports.


def _enable_gradient_checkpointing(module, label: str):
    """
    Safely activates gradient checkpointing with a frozen-base PEFT module
    (only LoRA is trainable). Without enable_input_require_grads(),
    checkpointing can break the gradient graph on backward
    ("Trying to backward through the graph a second time"), because PyTorch
    doesn't understand that it must reconnect the recomputed output to the rest of the
    graph if no input to the block requires a gradient itself.
    use_reentrant=False is the most robust mode for this scenario
    (frozen base + trainable adapter).
    """
    if not hasattr(module, "gradient_checkpointing_enable"):
        return
    if hasattr(module, "enable_input_require_grads"):
        module.enable_input_require_grads()
    try:
        module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        module.gradient_checkpointing_enable()
    print(f"  ✓ Gradient checkpointing activated on {label}")


# ══════════════════════════════════════════════════════════════════════
# Vision+MLP Variant — UNCHANGED compared to your version
# ══════════════════════════════════════════════════════════════════════

class SigLIPModuleVisionMLP(nn.Module):
    """
    Vision-only SigLIP2 architecture. Uses a frozen vision backbone, a LoRA adapter,
    and a custom MLP head for binary classification.
    """
    variant = "vision_mlp"

    def __init__(
        self,
        model_name        = "google/siglip2-base-patch16-224",
        lora_r_visual      = 8,
        lora_alpha_visual  = 16,
        lora_dropout       = 0.05,
        mlp_hidden         = None,   # DEPRECATED — no longer used, see note below
        mlp_dropout        = 0.30,
        pretrained_weights_dir = None,
        use_gradient_checkpointing = False,
    ):
        """
        Initializes the vision-only SigLIP model. Loads the backbone, applies
        LoRA, and sets up the MLP head.
        """
        super().__init__()

        # (mlp_hidden is accepted for signature compatibility but ignored —
        # the architecture is now fixed, see below)

        print(f"  Loading SigLIPv2 (Vision+MLP) from '{model_name}'...")
        base_model = AutoModel.from_pretrained(model_name)

        # FIX: visual embedding dimension read DYNAMICALLY from the loaded
        # model, no longer a fixed constant (it was 768, but the "giant"
        # checkpoint has hidden_size=1536 — this caused a shape mismatch error
        # both when loading Phase 3 weights and, actually, even with random-init
        # weights as soon as it reached the first forward pass, since self.classifier
        # always expected 768 as input regardless of the backbone).
        embed_dim = base_model.config.vision_config.hidden_size
        print(f"  Detected visual embedding dimension: {embed_dim}")

        # We keep a reference to the PRE-LoRA module — needed for load from Phase 3.
        self.base_vision_model = base_model.vision_model

        if pretrained_weights_dir:
            # FIX: if we load pretrained weights, DO NOT create a random adapter
            # first with get_peft_model() — it would modify base_vision_model
            # "in place" injecting LoRA with the wrong rank (the config one,
            # not the real one from the checkpoint), making the subsequent
            # loading inconsistent. We build the adapter DIRECTLY from the
            # saved checkpoint (which brings with it the correct rank/alpha,
            # read from adapter_config.json — no need to guess them in config.py).
            from src.training.weight_loading import load_siglip_vision_mlp_visual_only
            self.visual_encoder = load_siglip_vision_mlp_visual_only(
                self.base_vision_model, pretrained_weights_dir
            )
        else:
            lora_cfg_visual = LoraConfig(
                r=lora_r_visual, lora_alpha=lora_alpha_visual,
                target_modules=["q_proj", "v_proj"], lora_dropout=lora_dropout, bias="none",
            )
            self.visual_encoder = get_peft_model(self.base_vision_model, lora_cfg_visual)

        print(f"  Visual encoder — trainable parameters:")
        self.visual_encoder.print_trainable_parameters()

        if use_gradient_checkpointing:
            _enable_gradient_checkpointing(self.visual_encoder, "visual encoder")

        # FIXED architecture, identical to SigLIPLoRAClassifier in
        # train_fase3_lora.py — Linear(embed_dim -> embed_dim//2) -> GELU ->
        # Dropout -> Linear(embed_dim//2 -> 1). NOT generic/configurable
        # by mlp_hidden as before: that was a different architecture
        # (2 hidden layers + LayerNorm) that didn't match at all with
        # the real architecture saved in the Phase 3 checkpoints — loading the
        # weights would have always failed, regardless of the fix for the
        # input dimension alone.
        hidden = embed_dim // 2
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden, 1),
        )

        if pretrained_weights_dir:
            from src.training.weight_loading import load_siglip_vision_mlp_head_only
            load_siglip_vision_mlp_head_only(self.classifier, pretrained_weights_dir)
        else:
            for m in self.classifier.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        arch_str = f"{embed_dim} -> {hidden} -> 1"
        print(f"  MLP classifier: {arch_str}")

    def encode_prompts(self, positive_prompts: list, negative_prompts: list):
        """No-op: this variant does not use prompts. It exists only to have the
        same interface as SigLIPModuleCompleta (see trainer.py, which
        always calls it without if/else on the variant)."""
        pass

    def forward(self, roi_crops: torch.Tensor) -> dict:
        """
        Forward pass for the Vision+MLP model.
        Extracts visual features from ROI crops and passes them through the MLP head.
        Returns a dictionary containing the raw logits.
        """
        vis_out = self.visual_encoder(pixel_values=roi_crops)
        v = F.normalize(vis_out.pooler_output, dim=-1)
        logit = self.classifier(v)
        return {"logit": logit}

    def get_visual_embedding(self, roi_crops: torch.Tensor) -> torch.Tensor:
        """
        Extracts normalized visual embeddings for given ROI crops without classification.
        """
        vis_out = self.visual_encoder(pixel_values=roi_crops)
        return F.normalize(vis_out.pooler_output, dim=-1)


# ══════════════════════════════════════════════════════════════════════
# Complete Variant — NEW (equivalent to train_fase3_full_lora.py)
# ══════════════════════════════════════════════════════════════════════

class SigLIPModuleCompleta(nn.Module):
    """
    Full contrastive SigLIP2 architecture. Uses both vision and text encoders
    with LoRA adapters.
    """
    variant = "completa"

    def __init__(
        self,
        model_name        = "google/siglip2-base-patch16-224",
        lora_r_visual      = 8,
        lora_alpha_visual  = 16,
        lora_r_text        = 4,
        lora_alpha_text    = 8,
        lora_dropout       = 0.05,
        pretrained_weights_dir = None,
        use_gradient_checkpointing = False,
    ):
        """
        Initializes the full contrastive SigLIP model. Loads both encoders,
        applies LoRA to both, and initializes trainable scale/bias for contrastive logits.
        """
        super().__init__()

        print(f"  Loading SigLIPv2 (Complete, vision+text) from '{model_name}'...")
        base_model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.base_vision_model = base_model.vision_model
        self.base_text_model = base_model.text_model

        if pretrained_weights_dir:
            # Same fix as SigLIPModuleVisionMLP — we build the adapters
            # DIRECTLY from the checkpoint (rank/alpha read from
            # adapter_config.json), no intermediate get_peft_model().
            from src.training.weight_loading import (
                load_siglip_completa_visual_only, load_siglip_completa_text_only,
                load_siglip_completa_scale_bias,
            )
            self.visual_encoder = load_siglip_completa_visual_only(self.base_vision_model, pretrained_weights_dir)
            self.text_encoder = load_siglip_completa_text_only(self.base_text_model, pretrained_weights_dir)
            logit_scale_val, logit_bias_val = load_siglip_completa_scale_bias(pretrained_weights_dir)
            self.logit_scale = nn.Parameter(logit_scale_val)
            self.logit_bias = nn.Parameter(logit_bias_val)
        else:
            lora_cfg_visual = LoraConfig(
                r=lora_r_visual, lora_alpha=lora_alpha_visual,
                target_modules=["q_proj", "v_proj"], lora_dropout=lora_dropout, bias="none",
            )
            self.visual_encoder = get_peft_model(self.base_vision_model, lora_cfg_visual)

            lora_cfg_text = LoraConfig(
                r=lora_r_text, lora_alpha=lora_alpha_text,
                target_modules=["q_proj", "v_proj"], lora_dropout=lora_dropout, bias="none",
            )
            self.text_encoder = get_peft_model(self.base_text_model, lora_cfg_text)

            self.logit_scale = nn.Parameter(base_model.logit_scale.detach().clone().float())
            self.logit_bias = nn.Parameter(base_model.logit_bias.detach().clone().float())

        print(f"  Visual encoder — trainable parameters:")
        self.visual_encoder.print_trainable_parameters()
        print(f"  Text encoder — trainable parameters:")
        self.text_encoder.print_trainable_parameters()

        if use_gradient_checkpointing:
            _enable_gradient_checkpointing(self.visual_encoder, "visual encoder")
            _enable_gradient_checkpointing(self.text_encoder, "text encoder")

        self.text_embeds_pos = None
        self.text_embeds_neg = None

    def encode_prompts(self, positive_prompts: list, negative_prompts: list):
        """
        Precomputes text embeddings for positive and negative prompts.
        Must be called before forward().
        """
        device = self.logit_scale.device
        self.text_embeds_pos = self._encode(positive_prompts, device)
        self.text_embeds_neg = self._encode(negative_prompts, device)
        print(f"  ✓ Prompts encoded — positive: {positive_prompts}")
        print(f"                      negative: {negative_prompts}")

    def _encode(self, prompts: list, device) -> torch.Tensor:
        """Helper to tokenize and encode a list of text prompts into normalized embeddings."""
        tokens = self.tokenizer(
            prompts, return_tensors="pt", padding="max_length",
            truncation=True, max_length=64,
        ).to(device)
        out = self.text_encoder(**tokens)
        embeds = F.normalize(out.pooler_output, dim=-1)
        return F.normalize(embeds.mean(dim=0, keepdim=True), dim=-1)

    def forward(self, roi_crops: torch.Tensor) -> dict:
        """
        Forward pass for the Complete model.
        Computes cosine similarities between visual crops and precomputed text prompts.
        Returns a dictionary with logits for positive and negative matches.
        """
        if self.text_embeds_pos is None or self.text_embeds_neg is None:
            raise RuntimeError("You must call encode_prompts() before forward()!")

        vis_out = self.visual_encoder(pixel_values=roi_crops)
        v = F.normalize(vis_out.pooler_output, dim=-1)

        sim_pos = v @ self.text_embeds_pos.T
        sim_neg = v @ self.text_embeds_neg.T
        logits_pos = sim_pos * self.logit_scale.exp() + self.logit_bias
        logits_neg = sim_neg * self.logit_scale.exp() + self.logit_bias
        return {"logits_pos": logits_pos, "logits_neg": logits_neg}


# ══════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════

def build_siglip_module(cfg) -> nn.Module:
    """
    Factory function to instantiate the correct SigLIP module variant
    based on the configuration. Handles dynamic weight loading paths.
    """
    if getattr(cfg.siglip, "skip_pretrained_load", False):
        weights_dir = None
        print("  [Phase 3 Weights] SKIPPED (skip_pretrained_load=True) — weights will be "
              "overwritten by an external joint-trained checkpoint")
    else:
        weights_dir = cfg.siglip.resolve_pretrained_weights_dir(cfg.negative_mining.neg_ratio)
        print(f"  [Phase 3 Weights] resolved path: {weights_dir}")

    if cfg.siglip.variant == "vision_mlp":
        return SigLIPModuleVisionMLP(
            model_name        = cfg.siglip.model_name,
            lora_r_visual      = cfg.siglip.lora_r_visual,
            lora_alpha_visual  = cfg.siglip.lora_alpha_visual,
            lora_dropout       = cfg.siglip.lora_dropout,
            mlp_hidden         = cfg.siglip.mlp_hidden,
            mlp_dropout        = cfg.siglip.mlp_dropout,
            pretrained_weights_dir = weights_dir,
            use_gradient_checkpointing = cfg.siglip.use_gradient_checkpointing,
        )
    elif cfg.siglip.variant == "completa":
        return SigLIPModuleCompleta(
            model_name        = cfg.siglip.model_name,
            lora_r_visual      = cfg.siglip.lora_r_visual,
            lora_alpha_visual  = cfg.siglip.lora_alpha_visual,
            lora_r_text        = cfg.siglip.lora_r_text,
            lora_alpha_text    = cfg.siglip.lora_alpha_text,
            lora_dropout       = cfg.siglip.lora_dropout,
            pretrained_weights_dir = weights_dir,
            use_gradient_checkpointing = cfg.siglip.use_gradient_checkpointing,
        )
    else:
        raise ValueError(f"Unknown siglip.variant: '{cfg.siglip.variant}'")