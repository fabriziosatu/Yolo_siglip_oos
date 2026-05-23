import sys, torch
sys.path.insert(0, ".")
from src.data.dataset import build_dataloaders
from src.models.detector import YOLO26Detector

device = torch.device("cuda")
det = YOLO26Detector(
    "runs/detect/weights/phase1_yolo/weights/best.pt",
    conf_threshold=0.25
).to(device)

loader, _, _ = build_dataloaders(img_size=640, batch_size=2)
batch = next(iter(loader))
images = batch["images"].to(device)
boxes  = batch["boxes"]

gt0 = boxes[0].to(device)
xc,yc,w,h = gt0[:,0]*640, gt0[:,1]*640, gt0[:,2]*640, gt0[:,3]*640
gt_xyxy = torch.stack([xc-w/2, yc-h/2, xc+w/2, yc+h/2], dim=1)
print(f"GT box 0 in pixel: {gt_xyxy[0]}")

# EVAL MODE
det.eval()
with torch.no_grad():
    preds_eval, _, _ = det(images)
filtered_eval = det.filter_predictions(preds_eval)
if len(filtered_eval[0]) > 0:
    print(f"\nEVAL mode pred box 0: {filtered_eval[0][0,:4]}")
    print(f"  range x: [{filtered_eval[0][:,0].min():.1f}, {filtered_eval[0][:,2].max():.1f}]")
    print(f"  range y: [{filtered_eval[0][:,1].min():.1f}, {filtered_eval[0][:,3].max():.1f}]")

# TRAIN MODE
det.train()
with torch.no_grad():
    preds_train, _, raw = det(images)
filtered_train = det.filter_predictions(preds_train)
print(f"\nTRAIN mode preds shape: {preds_train.shape}")
print(f"  confidenze max: {preds_train[0,:,4].max().item():.4f}")
print(f"  box range x: [{preds_train[0,:,0].min():.3f}, {preds_train[0,:,2].max():.3f}]")
print(f"  box range y: [{preds_train[0,:,1].min():.3f}, {preds_train[0,:,3].max():.3f}]")
if len(filtered_train[0]) > 0:
    print(f"  pred box 0: {filtered_train[0][0,:4]}")

print(f"\nraw_detect in train mode:")
if isinstance(raw, dict):
    for k, v in raw.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, torch.Tensor):
                    print(f"  {k}.{k2}: shape={tuple(v2.shape)} range=[{v2.min():.3f}, {v2.max():.3f}]")