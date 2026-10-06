"""
src/models/detector.py
=======================
Wrapper around YOLO26 with fully differentiable predictions.

SOLUTION TO THE DETACH() PROBLEM:
  YOLO26 by default loads parameters with requires_grad=False and
  uses detach() internally in the Detect layer (x_detach = [xi.detach() for xi in x]).

  Two-step solution:
  1. Replaces the Detect layer with DetectNoDetach which removes the detach()
     ONLY in train mode — in eval mode the behavior is identical to the original
  2. Enables requires_grad=True on all model parameters

  Result: decoded boxes with grad_fn=True in absolute pixels [0,640]
  that allow a single backward pass on the joint loss.
  In eval mode NMS works correctly as in the original.

DIFFERENTIABLE DECODING:
  In train mode YOLO produces raw_boxes in undecoded DFL format.
  We use Ultralytics' decode_bboxes() + make_anchors() to decode
  into absolute pixels while keeping the computational graph intact.

  Confidence threshold for matching: 0.1
  -> during training we want to find candidate predictions
     for each GT box, not perform final inference
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.tal import make_anchors

FEATURE_LAYER_IDX = 13
FEATURE_CHANNELS  = 128
FEATURE_STRIDE    = 16

# Confidence threshold for matching during training
TRAIN_CONF_THRESHOLD = 0.1


class DetectNoDetach(Detect):
    """
    Subclass of Detect that removes the internal detach() ONLY in train mode.

    In train mode:
        # REMOVED the line with detach():
        # x_detach = [xi.detach() for xi in x]
        one2one = self.forward_head(x, **self.one2one)  <- direct x

    In eval mode:
        # IDENTICAL to the original — NMS works correctly
        x_detach = [xi.detach() for xi in x]
        one2one = self.forward_head(x_detach, **self.one2one)

    This ensures that:
    - In training: gradients flow through the one2one head
    - In eval: NMS receives data in the format expected by Ultralytics
    """

    def forward(self, x):
        """Forward pass for the modified Detect layer handling train/eval modes."""
        preds = self.forward_head(x, **self.one2many)
        if self.end2end:
            if self.training:
                # TRAIN MODE: removes detach() for differentiable graph
                one2one = self.forward_head(x, **self.one2one)
            else:
                # EVAL MODE: identical behavior to the original
                # detach() is necessary for Ultralytics NMS
                x_detach = [xi.detach() for xi in x]
                one2one  = self.forward_head(x_detach, **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)


class YOLO26Detector(nn.Module):
    """
    Wrapper for the YOLO26 model that modifies the detection head
    to support fully differentiable coordinate outputs during training.
    """

    def __init__(self, weights: str = "yolo26n.pt", conf_threshold: float = 0.25,
                 train_conf_threshold: float = None):
        """
        Initializes the YOLO26 wrapper, replaces the detection head, and sets up
        gradient hooks for feature extraction.
        """
        super().__init__()
        self.conf_threshold       = conf_threshold
        # If not specified, uses the historical default (0.1 — designed for YOLO
        # not yet calibrated, e.g. init_mode="scratch"). With Phase 1 weights already
        # well trained (init_mode="pretrained") a higher value is better
        # (e.g. 0.2) — less "noise" ROIs to pass to SigLIP, less memory,
        # see discussion on TRAIN_CONF_THRESHOLD and OOM.
        self.train_conf_threshold = (train_conf_threshold if train_conf_threshold is not None
                                      else TRAIN_CONF_THRESHOLD)
        self.weights              = weights

        yolo = YOLO(weights)
        self.model = yolo.model

        # ── Step 1: replace Detect with DetectNoDetach ────────────────────
        old_detect = self.model.model[-1]
        new_detect = DetectNoDetach.__new__(DetectNoDetach)
        new_detect.__dict__.update(old_detect.__dict__)
        new_detect.__class__ = DetectNoDetach
        self.model.model[-1] = new_detect

        # ── Step 2: enable requires_grad on all parameters ───────────────
        for p in self.model.parameters():
            p.requires_grad_(True)

        # ── Step 3: freeze the Detect layer ─────────────────────────────────
        # Preserves calibrated anchors and strides — needed for decoding
        for p in self.model.model[-1].parameters():
            p.requires_grad_(False)

        self.model.train()

        # ── Hook: feature map from layer 13 (for RoI Align) ───────────────────
        self._feature_map = None
        self._hook_feat   = self.model.model[FEATURE_LAYER_IDX].register_forward_hook(
            lambda m, i, o: setattr(self, "_feature_map", o)
        )

    def forward(self, images: torch.Tensor):
        """
        Forward pass with differentiable predictions.

        Train mode:
          - Captures feature map from layer 13 (with grad_fn)
          - Decodes boxes into absolute pixels keeping grad_fn
          - Returns differentiable predictions (B, N, 6)

        Eval mode:
          - Uses standard Ultralytics forward with correct NMS
          - DetectNoDetach restores detach() in eval mode
          - Returns accurate post-NMS predictions

        Returns:
            predictions : (B, N, 6) [x1,y1,x2,y2,conf,cls] in pixels
            feature_map : (B, 128, 40, 40) with grad_fn in train, without in eval
            raw_detect  : dict with one2one, one2many
        """
        self._feature_map = None

        if self.training:
            # Forward in train mode — DetectNoDetach keeps grad_fn
            raw_out     = self.model(images)
            feature_map = self._feature_map
            predictions = self._decode_train_predictions(raw_out, images.device)
            return predictions, feature_map, raw_out

        else:
            # Eval mode — NMS correct thanks to restored detach()
            with torch.no_grad():
                eval_out    = self.model(images)
            feature_map = self._feature_map

            if isinstance(eval_out, (list, tuple)):
                predictions = eval_out[0]
            else:
                predictions = torch.zeros(
                    images.shape[0], 300, 6, device=images.device
                )

            return predictions, feature_map, eval_out

    def _decode_train_predictions(
        self, raw_out: dict, device: torch.device
    ) -> torch.Tensor:
        """
        Decodes train mode predictions into absolute pixels.

        Uses Ultralytics decode_bboxes() and make_anchors() to
        convert raw_boxes into xyxy coordinates [0,640],
        keeping the computational graph for the backward pass.

        Returns:
            (B, N, 6) tensor with [x1,y1,x2,y2,conf,cls] and grad_fn
        """
        detect    = self.model.model[-1]
        one2one   = raw_out["one2one"]
        raw_boxes = one2one["boxes"]    # (B, 4, 8400)
        scores    = one2one["scores"]   # (B, nc, 8400)
        feats     = one2one["feats"]    # list of 3 feature maps

        # Calculate anchors and strides from feature levels
        anchors, strides = (
            a.transpose(0, 1)
            for a in make_anchors(feats, detect.stride, 0.5)
        )

        # Decode: raw_boxes -> absolute pixels (keeps grad_fn)
        decoded = detect.decode_bboxes(
            detect.dfl(raw_boxes),
            anchors.unsqueeze(0)
        ) * strides
        # decoded shape: (B, 4, 8400) in xyxy format [0,640]

        # Confidence: max score among classes
        conf = scores.sigmoid().max(dim=1, keepdim=True).values  # (B,1,8400)

        # Predicted class
        cls  = scores.sigmoid().argmax(dim=1, keepdim=True).float()  # (B,1,8400)

        # Assemble output (B, 8400, 6): [x1,y1,x2,y2,conf,cls]
        predictions = torch.cat([
            decoded.permute(0, 2, 1),   # (B, 8400, 4)
            conf.permute(0, 2, 1),      # (B, 8400, 1)
            cls.permute(0, 2, 1),       # (B, 8400, 1)
        ], dim=2)                        # (B, 8400, 6)

        return predictions

    def filter_predictions(self, predictions: torch.Tensor):
        """
        Filters predictions by confidence threshold.

        In train mode uses a low threshold (0.1) for IoU matching.
        In eval mode uses the standard threshold (0.25).
        """
        threshold = (self.train_conf_threshold
                     if self.training else self.conf_threshold)

        filtered = []
        for pred in predictions:
            with torch.no_grad():
                mask = pred[:, 4] >= threshold
            # keep grad_fn on filtered predictions
            filtered.append(pred[mask])
        return filtered

    def predictions_to_roi_format(self, filtered_preds: list, batch_size: int):
        """Converts predictions into the format [batch_idx, x1, y1, x2, y2]."""
        roi_list = []
        for batch_idx, preds in enumerate(filtered_preds):
            if len(preds) == 0:
                continue
            idx_col = torch.full(
                (len(preds), 1), batch_idx,
                dtype=torch.float32, device=preds.device
            )
            roi_list.append(torch.cat([idx_col, preds[:, :4]], dim=1))

        if roi_list:
            return torch.cat(roi_list, dim=0)
        device = filtered_preds[0].device if filtered_preds else torch.device("cpu")
        return torch.zeros((0, 5), dtype=torch.float32, device=device)

    def remove_hooks(self):
        """Removes PyTorch forward hooks safely to prevent memory leaks."""
        self._hook_feat.remove()

    def __del__(self):
        try:
            self.remove_hooks()
        except Exception:
            pass


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    print("Test YOLO26Detector differentiable...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from src.utils.config import CFG, PHASE1_WEIGHTS
    det = YOLO26Detector(str(PHASE1_WEIGHTS), conf_threshold=0.25).to(device)

    # Test train mode
    det.train()
    from src.data.dataset import build_dataloaders
    loader, _, _ = build_dataloaders(
        img_size=640, batch_size=2,
        data_dir=str(CFG.data.data_dir), num_workers=0
    )
    batch  = next(iter(loader))
    images = batch["images"].to(device)
    boxes  = batch["boxes"]

    preds, fmap, raw = det(images)
    print(f"[TRAIN] predictions shape : {preds.shape}")
    print(f"[TRAIN] predictions grad_fn: {preds.grad_fn is not None}")
    print(f"[TRAIN] feature_map grad_fn: {fmap.grad_fn is not None}")

    filtered = det.filter_predictions(preds)
    print(f"[TRAIN] Predictions above threshold {det.train_conf_threshold}:")
    for i, fp in enumerate(filtered):
        print(f"  img {i}: {len(fp)} predictions")

    from torchvision.ops import box_iou
    gt = boxes[0].to(device)
    xc,yc,w,h = gt[:,0]*640, gt[:,1]*640, gt[:,2]*640, gt[:,3]*640
    gt_xyxy = torch.stack([xc-w/2, yc-h/2, xc+w/2, yc+h/2], dim=1)
    if len(filtered[0]) > 0:
        with torch.no_grad():
            iou = box_iou(filtered[0][:,:4], gt_xyxy)
        print(f"[TRAIN] Max IoU with GT: {iou.max().item():.4f}")

    # Test eval mode
    det.eval()
    preds_eval, fmap_eval, _ = det(images)
    print(f"\n[EVAL] predictions shape : {preds_eval.shape}")
    print(f"[EVAL] n predictions img0 : {len(det.filter_predictions(preds_eval)[0])}")

    print("\n✓ YOLO26Detector OK!")