import sys, torch
sys.path.insert(0, '.')

def main():
    from ultralytics import YOLO
    from ultralytics.utils.tal import make_anchors
    from torchvision.ops import box_iou
    from src.data.dataset import build_dataloaders

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolo  = YOLO('weights/phase1_yolo/best.pt')
    model = yolo.model.to(device)
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)

    loader, _, _ = build_dataloaders(img_size=640, batch_size=2, num_workers=0)
    batch  = next(iter(loader))
    images = batch['images'].to(device)
    boxes  = batch['boxes']

    out = model(images)
    detect    = model.model[-1]
    raw_boxes = out['one2one']['boxes']
    scores    = out['one2one']['scores']
    feats     = out['one2one']['feats']

    anchors, strides = (a.transpose(0,1) for a in
                        make_anchors(feats, detect.stride, 0.5))

    decoded = detect.decode_bboxes(
        detect.dfl(raw_boxes), anchors.unsqueeze(0)
    ) * strides

    # decoded shape: (B, 4, 8400) — formato xyxy per end2end=True
    pred = decoded[0].permute(1, 0)   # (8400, 4)
    conf = scores[0].sigmoid().max(dim=0).values  # (8400,)

    # GT in pixel
    gt = boxes[0].to(device)
    xc,yc,w,h = gt[:,0]*640, gt[:,1]*640, gt[:,2]*640, gt[:,3]*640
    gt_xyxy = torch.stack([xc-w/2, yc-h/2, xc+w/2, yc+h/2], dim=1)

    print(f'decoded grad_fn: {decoded.grad_fn is not None}')
    print(f'GT boxes: {gt_xyxy}')
    print()

    # Filtra per confidenza e coordinate valide
    for conf_thr in [0.001, 0.01, 0.1, 0.25]:
        mask = (conf >= conf_thr) & \
               (pred[:,0] >= 0) & (pred[:,1] >= 0) & \
               (pred[:,2] <= 640) & (pred[:,3] <= 640)
        n = mask.sum().item()
        if n == 0:
            print(f'conf>={conf_thr}: 0 predizioni valide')
            continue
        pred_filt = pred[mask]
        with torch.no_grad():
            iou = box_iou(pred_filt, gt_xyxy)
        print(f'conf>={conf_thr}: {n} pred | '
              f'max IoU={iou.max().item():.4f} | '
              f'IoU>0.5={( iou.max(dim=1).values>0.5).sum().item()} | '
              f'IoU>0.3={(iou.max(dim=1).values>0.3).sum().item()}')

    # Confronto con eval mode
    print('\n--- Confronto eval mode ---')
    model.eval()
    with torch.no_grad():
        out_eval = model(images)
    if isinstance(out_eval, tuple):
        preds_eval = out_eval[0]  # (B, N, 6)
    else:
        preds_eval = out_eval
    pred_eval = preds_eval[0]
    mask_eval = pred_eval[:, 4] >= 0.25
    pred_eval_filt = pred_eval[mask_eval]
    if len(pred_eval_filt) > 0:
        iou_eval = box_iou(pred_eval_filt[:, :4], gt_xyxy)
        print(f'eval mode conf>=0.25: {len(pred_eval_filt)} pred | '
              f'max IoU={iou_eval.max().item():.4f} | '
              f'IoU>0.5={(iou_eval.max(dim=1).values>0.5).sum().item()}')

if __name__ == '__main__':
    main()