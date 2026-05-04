"""
train.py
========
Entraîne un TinyDetector avec l'opérateur de convolution spécifié.

Usage (Google Colab / Kaggle) :
    python train.py --conv sac --data /path/to/visdrone --epochs 50

Arguments :
    --conv      : nom de l'opérateur (voir CONV_REGISTRY)
    --data      : dossier racine VisDrone
    --epochs    : nombre d'epochs (défaut 50)
    --batch     : taille du batch (défaut 8)
    --img_size  : taille des images (défaut 640)
    --base_ch   : largeur du backbone (défaut 32)
    --lr        : learning rate initial (défaut 1e-3)
    --out_dir   : dossier de sortie (défaut ./runs)
    --workers   : num_workers dataloader (défaut 2)
    --device    : 'cuda' ou 'cpu' (auto-détecté si absent)
"""

import os
import sys
import time
import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Ajoute le dossier parent au path (pour import relatifs en Colab)
sys.path.insert(0, str(Path(__file__).parent))

from models import build_detector, CONV_REGISTRY
from data.visdrone import build_dataloaders, NUM_CLASSES
from utils import MAPMetric, decode_predictions, setup_logger, CSVLogger


# ─────────────────────────────────────────────────────────────────────────────
# Perte de détection
# ─────────────────────────────────────────────────────────────────────────────
class DetectionLoss(torch.nn.Module):
    """
    Perte simplifiée pour le détecteur léger :
        L = λ_cls * BCE(cls_preds, cls_targets)
          + λ_reg * SmoothL1(reg_preds, reg_targets)

    On assigne chaque vérité terrain à l'ancre la plus proche (grille uniforme).
    """

    def __init__(self, num_classes: int = 10,
                 lambda_cls: float = 1.0,
                 lambda_reg: float = 5.0):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_cls = lambda_cls
        self.lambda_reg = lambda_reg
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.sl1 = torch.nn.SmoothL1Loss(reduction='mean')

    def forward(self, cls_preds, reg_preds, boxes, mask, img_size=640):
        """
        Args:
            cls_preds : list[(B, C, H', W')]
            reg_preds : list[(B, 4, H', W')]
            boxes     : (B, max_dets, 5)  xywh_norm + cls
            mask      : (B, max_dets)     bool
        """
        total_cls = torch.tensor(0.0, device=cls_preds[0].device)
        total_reg = torch.tensor(0.0, device=cls_preds[0].device)
        n_levels  = 0

        for cls_map, reg_map in zip(cls_preds, reg_preds):
            B, C, H, W = cls_map.shape

            # Cibles classification : (B, C, H, W) — zéros par défaut
            cls_tgt = torch.zeros_like(cls_map)
            reg_tgt = torch.zeros_like(reg_map)
            has_reg = torch.zeros(B, 1, H, W,
                                  device=cls_map.device, dtype=torch.bool)

            for b in range(B):
                valid_boxes = boxes[b][mask[b]]    # (N_valid, 5)
                if len(valid_boxes) == 0:
                    continue

                # Assigne chaque GT à la cellule de la grille qui contient son centre
                xc = (valid_boxes[:, 0] * W).long().clamp(0, W-1)
                yc = (valid_boxes[:, 1] * H).long().clamp(0, H-1)
                cls_ids = valid_boxes[:, 4].long()

                for i in range(len(valid_boxes)):
                    ci, ri = cls_ids[i], (yc[i], xc[i])
                    cls_tgt[b, ci, ri[0], ri[1]] = 1.0
                    # Offset par rapport à la cellule
                    reg_tgt[b, 0, ri[0], ri[1]] = valid_boxes[i, 0] * W - xc[i]
                    reg_tgt[b, 1, ri[0], ri[1]] = valid_boxes[i, 1] * H - yc[i]
                    reg_tgt[b, 2, ri[0], ri[1]] = (valid_boxes[i, 2] * W).log().clamp(-4, 4)
                    reg_tgt[b, 3, ri[0], ri[1]] = (valid_boxes[i, 3] * H).log().clamp(-4, 4)
                    has_reg[b, 0, ri[0], ri[1]] = True

            loss_cls = self.bce(cls_map, cls_tgt)

            # Régression uniquement sur les cellules positives
            if has_reg.any():
                loss_reg = self.sl1(reg_map[has_reg.expand_as(reg_map)],
                                    reg_tgt[has_reg.expand_as(reg_tgt)])
            else:
                loss_reg = torch.tensor(0.0, device=cls_map.device)

            total_cls = total_cls + loss_cls
            total_reg = total_reg + loss_reg
            n_levels += 1

        n = max(n_levels, 1)
        return (self.lambda_cls * total_cls / n +
                self.lambda_reg * total_reg / n)


# ─────────────────────────────────────────────────────────────────────────────
# Boucle d'entraînement
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, img_size):
    model.train()
    total_loss = 0.0
    for batch in loader:
        imgs   = batch['image'].to(device)
        boxes  = batch['boxes'].to(device)
        mask   = batch['mask'].to(device)

        optimizer.zero_grad()
        cls_preds, reg_preds = model(imgs)
        loss = criterion(cls_preds, reg_preds, boxes, mask, img_size)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# ─────────────────────────────────────────────────────────────────────────────
# Boucle de validation
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, metric, device, img_size):
    model.eval()
    metric.reset()

    for batch in loader:
        imgs  = batch['image'].to(device)
        boxes = batch['boxes']    # CPU
        mask  = batch['mask']     # CPU

        cls_preds, reg_preds = model(imgs)

        # Décode les prédictions
        preds = decode_predictions(cls_preds, reg_preds,
                                   img_size=img_size, conf_thresh=0.01,
                                   num_classes=NUM_CLASSES)

        # Prépare les GT pour la métrique (format x1y1x2y2 + cls)
        gts = []
        for b in range(len(preds)):
            valid = boxes[b][mask[b]]          # (N,5) xywh_norm+cls
            if len(valid) == 0:
                gts.append(torch.zeros(0, 5))
                continue
            # xywh → xyxy
            gt_xyxy = torch.zeros_like(valid)
            gt_xyxy[:, 0] = valid[:, 0] - valid[:, 2] / 2
            gt_xyxy[:, 1] = valid[:, 1] - valid[:, 3] / 2
            gt_xyxy[:, 2] = valid[:, 0] + valid[:, 2] / 2
            gt_xyxy[:, 3] = valid[:, 1] + valid[:, 3] / 2
            gt_xyxy[:, 4] = valid[:, 4]
            gts.append(gt_xyxy)

        metric.update(preds, gts)

    return metric.compute()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="SAC Experiment — VisDrone")
    p.add_argument('--conv',     type=str, default='sac',
                   choices=list(CONV_REGISTRY),
                   help='Opérateur de convolution')
    p.add_argument('--data',     type=str, required=True,
                   help='Dossier racine VisDrone')
    p.add_argument('--epochs',   type=int, default=50)
    p.add_argument('--batch',    type=int, default=8)
    p.add_argument('--img_size', type=int, default=640)
    p.add_argument('--base_ch',  type=int, default=32)
    p.add_argument('--lr',       type=float, default=1e-3)
    p.add_argument('--out_dir',  type=str, default='./runs')
    p.add_argument('--workers',  type=int, default=2)
    p.add_argument('--device',   type=str, default='')
    return p.parse_args()


def main():
    args = parse_args()

    # ── Device ───────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    # ── Dossiers de sortie ───────────────────────────────────────────────────
    run_dir = Path(args.out_dir) / args.conv
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / 'best.pt'

    logger    = setup_logger(args.conv, str(run_dir), args.conv)
    csv_log   = CSVLogger(str(run_dir / 'metrics.csv'))

    logger.info(f"Convolution : {args.conv.upper()}")
    logger.info(f"Device      : {device}")
    logger.info(f"Epochs      : {args.epochs} | Batch : {args.batch} | "
                f"img_size : {args.img_size}")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_dl, val_dl = build_dataloaders(
        args.data, img_size=args.img_size,
        batch_size=args.batch, num_workers=args.workers
    )

    # ── Modèle ───────────────────────────────────────────────────────────────
    model = build_detector(args.conv, num_classes=NUM_CLASSES,
                           base_ch=args.base_ch).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Paramètres  : {n_params:,}")

    # ── Optimiseur & scheduler ───────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = DetectionLoss(num_classes=NUM_CLASSES)
    metric    = MAPMetric(num_classes=NUM_CLASSES)

    # ── Boucle ───────────────────────────────────────────────────────────────
    best_map50 = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_dl, optimizer, criterion, device, args.img_size
        )
        val_results = validate(
            model, val_dl, metric, device, args.img_size
        )
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]
        map50   = val_results['mAP50']
        map5095 = val_results['mAP50_95']

        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss={train_loss:.4f} | "
            f"mAP@50={map50:.4f} | "
            f"mAP@50:95={map5095:.4f} | "
            f"lr={lr_now:.2e} | "
            f"{elapsed:.0f}s"
        )
        csv_log.log(epoch, train_loss, map50, map5095, lr_now, elapsed)

        # Sauvegarde meilleur modèle
        if map50 > best_map50:
            best_map50 = map50
            torch.save({
                'epoch':      epoch,
                'conv':       args.conv,
                'state_dict': model.state_dict(),
                'mAP50':      map50,
                'mAP50_95':   map5095,
                'n_params':   n_params,
            }, ckpt_path)
            logger.info(f"  ✓ Nouveau meilleur mAP@50 = {map50:.4f} "
                        f"→ sauvegardé dans {ckpt_path}")

    logger.info(f"\nEntraînement terminé. Meilleur mAP@50 = {best_map50:.4f}")


if __name__ == '__main__':
    main()
