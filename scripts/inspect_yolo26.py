"""
scripts/inspect_yolo26.py
==========================
Esplora la struttura interna di YOLO26n per identificare
il layer corretto da cui estrarre la feature map per RoI Align.

Esegui con:
    python scripts/inspect_yolo26.py
"""

import torch
from ultralytics import YOLO

print("Carico YOLO26n...")
m = YOLO("yolo26n.pt")
model = m.model
model.eval()

# ── 1. Struttura dei layer interni (model.model è una Sequential) ─────────────
print("\n=== LAYER INTERNI (model.model) ===")
for i, layer in enumerate(model.model):
    print(f"  [{i:2d}] {type(layer).__name__:<30} | f={getattr(layer,'f', '?')}")

# ── 2. Forward hook: vediamo le shape delle feature map a ogni layer ──────────
print("\n=== SHAPE FEATURE MAP A OGNI LAYER ===")
shapes = {}

def make_hook(idx):
    def hook(module, inp, out):
        if isinstance(out, torch.Tensor):
            shapes[idx] = tuple(out.shape)
        elif isinstance(out, (list, tuple)):
            shapes[idx] = [tuple(x.shape) for x in out if isinstance(x, torch.Tensor)]
    return hook

hooks = []
for i, layer in enumerate(model.model):
    hooks.append(layer.register_forward_hook(make_hook(i)))

# Esegui un forward pass con immagine dummy
dummy = torch.zeros(1, 3, 640, 640)
with torch.no_grad():
    try:
        _ = model(dummy)
    except Exception as e:
        print(f"  (forward diretto fallito: {e}, provo con model.model)")
        try:
            _ = model.model(dummy)
        except Exception as e2:
            print(f"  Anche model.model fallito: {e2}")

for h in hooks:
    h.remove()

for idx, shape in shapes.items():
    print(f"  layer [{idx:2d}]: {shape}")

# ── 3. Identifica i layer candidati per RoI Align ────────────────────────────
print("\n=== LAYER CANDIDATI PER ROI ALIGN ===")
print("Cerco feature map con stride 16 o 32 (risoluzione 40x40 o 20x20)...")
for idx, shape in shapes.items():
    if isinstance(shape, tuple) and len(shape) == 4:
        h, w = shape[2], shape[3]
        if h in (20, 40) and w in (20, 40):
            stride = 640 // h
            print(f"  ✓ layer [{idx:2d}]: shape={shape} | stride={stride}x")

print("\nDone. Copia e incolla l'output completo e mandamelo.")