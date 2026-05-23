"""
scripts/debug_yolo_loss.py
===========================
Verifica che l'hook sul Detect layer catturi l'output corretto
e che il criterion interno di YOLO26 sia accessibile.

Esegui con:
    python scripts/debug_yolo_loss.py
"""

import sys
sys.path.insert(0, ".")

import torch
from ultralytics import YOLO

print("=" * 60)
print("Debug YOLO26 loss nativa")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Carica il modello
yolo = YOLO("yolo26n.pt")
model = yolo.model.to(device)
model.train()

# ── 1. Verifica che model.loss esista ────────────────────────────────────────
print("[1] Verifica model.loss...")
if hasattr(model, "loss"):
    print("  ✓ model.loss esiste")
else:
    print("  ✗ model.loss NON esiste — provo model.criterion...")
    if hasattr(model, "criterion"):
        print("  ✓ model.criterion esiste")
    else:
        print("  ✗ nemmeno model.criterion — lista attributi disponibili:")
        attrs = [a for a in dir(model) if not a.startswith("_")]
        print(f"  {attrs}")

# ── 2. Registra hook sul Detect layer ────────────────────────────────────────
print("\n[2] Registro hook sul Detect layer (model.model[-1])...")
raw_output_captured = {}

def hook_fn(module, input, output):
    raw_output_captured["output"] = output
    raw_output_captured["type"]   = type(output).__name__
    if isinstance(output, torch.Tensor):
        raw_output_captured["shape"] = tuple(output.shape)
    elif isinstance(output, (list, tuple)):
        raw_output_captured["shapes"] = [
            tuple(x.shape) if isinstance(x, torch.Tensor) else type(x).__name__
            for x in output
        ]
    elif isinstance(output, dict):
        raw_output_captured["keys"] = list(output.keys())

hook = model.model[-1].register_forward_hook(hook_fn)

# ── 3. Forward pass ──────────────────────────────────────────────────────────
print("[3] Forward pass con immagine dummy...")
dummy = torch.rand(2, 3, 640, 640).to(device)

with torch.no_grad():
    _ = model(dummy)

hook.remove()

print(f"  Tipo output catturato: {raw_output_captured.get('type', 'N/A')}")
if "shape" in raw_output_captured:
    print(f"  Shape: {raw_output_captured['shape']}")
if "shapes" in raw_output_captured:
    print(f"  Shapes: {raw_output_captured['shapes']}")
if "keys" in raw_output_captured:
    print(f"  Chiavi dict: {raw_output_captured['keys']}")

# ── 4. Prova a chiamare model.loss ───────────────────────────────────────────
print("\n[4] Provo a chiamare model.loss con batch fittizio...")

# Costruisci batch GT fittizio
batch = {
    "img":       dummy,
    "cls":       torch.zeros(4, device=device),          # 4 box totali
    "bboxes":    torch.rand(4, 4, device=device),        # 4 box normalizzate
    "batch_idx": torch.tensor([0,0,1,1], dtype=torch.float32, device=device),
}

raw = raw_output_captured.get("output", None)

if raw is not None and hasattr(model, "loss"):
    try:
        loss, loss_items = model.loss(batch, raw)
        print(f"  ✓ model.loss(batch, raw) OK — loss={loss.item():.4f}")
        print(f"  loss_items: {loss_items}")
    except Exception as e1:
        print(f"  ✗ model.loss(batch, raw) fallito: {e1}")
        try:
            loss, loss_items = model.loss(raw, batch)
            print(f"  ✓ model.loss(raw, batch) OK — loss={loss.item():.4f}")
        except Exception as e2:
            print(f"  ✗ anche model.loss(raw, batch) fallito: {e2}")

            # Prova a ispezionare la firma del metodo
            import inspect
            try:
                sig = inspect.signature(model.loss)
                print(f"\n  Firma di model.loss: {sig}")
            except Exception:
                pass
else:
    if raw is None:
        print("  ✗ Hook non ha catturato nulla")
    else:
        print("  ✗ model.loss non disponibile")

print("\n" + "=" * 60)
print("Fine debug")