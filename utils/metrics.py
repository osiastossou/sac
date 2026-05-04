"""
metrics.py
==========
Calcul du mAP@50 et mAP@50:95 pour la détection d'objets.

Implémentation from scratch compatible CPU/GPU, sans dépendance à pycocotools.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict


# ─────────────────────────────────────────────────────────────────────────────
# IoU
# ─────────────────────────────────────────────────────────────────────────────
def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    IoU entre deux ensembles de boîtes au format [x1,y1,x2,y2].

    Args:
        boxes1 : (N, 4)
        boxes2 : (M, 4)
    Returns:
        iou    : (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-8)


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """[xc, yc, w, h] → [x1, y1, x2, y2]"""
    b = boxes.clone()
    b[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    b[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    b[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
    b[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
    return b


# ─────────────────────────────────────────────────────────────────────────────
# AP par classe
# ─────────────────────────────────────────────────────────────────────────────
def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Calcul de l'AP par interpolation sur 101 points (méthode COCO).
    """
    recalls    = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([1.0], precisions, [0.0]))

    # Cumul max inverse
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Intégrale sur 101 seuils
    thresholds = np.linspace(0, 1, 101)
    ap = 0.0
    for t in thresholds:
        mask = recalls >= t
        if mask.any():
            ap += precisions[mask].max()
    return ap / 101.0


# ─────────────────────────────────────────────────────────────────────────────
# Décodage des prédictions du détecteur
# ─────────────────────────────────────────────────────────────────────────────
def decode_predictions(cls_preds: List[torch.Tensor],
                       reg_preds: List[torch.Tensor],
                       img_size: int,
                       conf_thresh: float = 0.01,
                       num_classes: int = 10
                       ) -> List[torch.Tensor]:
    """
    Convertit les sorties du détecteur (listes de feature maps) en listes de
    détections par image.

    Args:
        cls_preds   : list[(B, num_classes, H', W')] — logits de classification
        reg_preds   : list[(B, 4, H', W')]           — prédictions de boîtes
        img_size    : taille de l'image d'entrée
        conf_thresh : seuil de confiance minimal
        num_classes : nombre de classes

    Returns:
        detections : list[Tensor(N_det, 6)] avec colonnes [x1,y1,x2,y2,conf,cls]
                     une entrée par image dans le batch
    """
    B = cls_preds[0].shape[0]
    all_dets = [[] for _ in range(B)]

    for cls_map, reg_map in zip(cls_preds, reg_preds):
        _, _, H, W = cls_map.shape
        stride_h = img_size / H
        stride_w = img_size / W

        # Grille de centres des ancres
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=cls_map.device),
            torch.arange(W, dtype=torch.float32, device=cls_map.device),
            indexing='ij'
        )
        cx = (grid_x + 0.5) * stride_w / img_size   # normalisé [0,1]
        cy = (grid_y + 0.5) * stride_h / img_size

        scores = cls_map.sigmoid()                   # (B, C, H, W)
        conf, cls_id = scores.max(dim=1)             # (B, H, W) chacun

        # Boîtes prédites (format xywh normalisé)
        bx = reg_map[:, 0].sigmoid() + cx.unsqueeze(0)
        by = reg_map[:, 1].sigmoid() + cy.unsqueeze(0)
        bw = reg_map[:, 2].exp() * (stride_w / img_size)
        bh = reg_map[:, 3].exp() * (stride_h / img_size)

        # Aplatit les positions
        conf   = conf.view(B, -1)
        cls_id = cls_id.view(B, -1).float()
        bx = bx.view(B, -1); by = by.view(B, -1)
        bw = bw.view(B, -1); bh = bh.view(B, -1)

        for b in range(B):
            mask = conf[b] > conf_thresh
            if not mask.any():
                continue
            det = torch.stack([
                (bx[b][mask] - bw[b][mask] / 2).clamp(0, 1),
                (by[b][mask] - bh[b][mask] / 2).clamp(0, 1),
                (bx[b][mask] + bw[b][mask] / 2).clamp(0, 1),
                (by[b][mask] + bh[b][mask] / 2).clamp(0, 1),
                conf[b][mask],
                cls_id[b][mask],
            ], dim=1)                               # (N, 6)
            all_dets[b].append(det)

    # Concatène les détections de chaque niveau FPN
    result = []
    for b in range(B):
        if all_dets[b]:
            result.append(torch.cat(all_dets[b], dim=0))
        else:
            result.append(torch.zeros(0, 6))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# mAP
# ─────────────────────────────────────────────────────────────────────────────
class MAPMetric:
    """
    Accumule les prédictions et les vérités terrain, puis calcule
    mAP@50 et mAP@50:95 à la fin de l'epoch.

    Usage :
        metric = MAPMetric(num_classes=10)
        for batch in val_loader:
            ...
            metric.update(predictions, ground_truths)
        results = metric.compute()
        metric.reset()
    """

    def __init__(self, num_classes: int = 10, iou_thresholds: list = None):
        self.num_classes = num_classes
        if iou_thresholds is None:
            # COCO : de 0.50 à 0.95 par pas de 0.05
            self.iou_thresholds = np.arange(0.50, 1.00, 0.05).tolist()
        else:
            self.iou_thresholds = iou_thresholds

        self.predictions: List[torch.Tensor] = []   # (N,6) x1y1x2y2 conf cls
        self.ground_truths: List[torch.Tensor] = [] # (M,5) x1y1x2y2 cls

    def reset(self):
        self.predictions.clear()
        self.ground_truths.clear()

    def update(self,
               preds: List[torch.Tensor],
               gts:   List[torch.Tensor]):
        """
        Args:
            preds : list[Tensor(N,6)] — une entrée par image
            gts   : list[Tensor(M,5)] — une entrée par image
        """
        for p, g in zip(preds, gts):
            self.predictions.append(p.detach().cpu())
            self.ground_truths.append(g.detach().cpu())

    def compute(self) -> Dict[str, float]:
        """
        Calcule mAP@50 et mAP@50:95 sur toutes les images accumulées.

        Returns:
            {'mAP50': float, 'mAP50_95': float, 'per_class_AP50': dict}
        """
        ap_per_iou = []

        for iou_thresh in self.iou_thresholds:
            ap_per_class = []
            for cls in range(self.num_classes):
                tp_list, conf_list, n_gt = [], [], 0

                for pred, gt in zip(self.predictions, self.ground_truths):
                    # Filtrer par classe
                    gt_cls  = gt[gt[:, 4] == cls, :4] if len(gt) else torch.zeros(0,4)
                    pred_cls = pred[pred[:, 5] == cls] if len(pred) else torch.zeros(0,6)
                    n_gt += len(gt_cls)

                    if len(pred_cls) == 0:
                        continue

                    # Trier par confiance décroissante
                    order   = pred_cls[:, 4].argsort(descending=True)
                    pred_cls = pred_cls[order]

                    if len(gt_cls) == 0:
                        tp_list.append(torch.zeros(len(pred_cls)))
                        conf_list.append(pred_cls[:, 4])
                        continue

                    iou = box_iou_xyxy(pred_cls[:, :4], gt_cls)
                    matched = torch.zeros(len(gt_cls), dtype=torch.bool)
                    tp = torch.zeros(len(pred_cls))

                    for i in range(len(pred_cls)):
                        iou_row = iou[i]
                        best_iou, best_j = iou_row.max(0)
                        if best_iou >= iou_thresh and not matched[best_j]:
                            tp[i] = 1
                            matched[best_j] = True

                    tp_list.append(tp)
                    conf_list.append(pred_cls[:, 4])

                if not tp_list or n_gt == 0:
                    ap_per_class.append(0.0)
                    continue

                tp_all   = torch.cat(tp_list).numpy()
                conf_all = torch.cat(conf_list).numpy()
                order    = conf_all.argsort()[::-1]
                tp_all   = tp_all[order]

                cum_tp = tp_all.cumsum()
                cum_fp = (1 - tp_all).cumsum()
                recalls    = cum_tp / (n_gt + 1e-8)
                precisions = cum_tp / (cum_tp + cum_fp + 1e-8)

                ap_per_class.append(compute_ap(recalls, precisions))

            ap_per_iou.append(np.mean(ap_per_class))

        map50    = ap_per_iou[0]                    # IoU=0.50
        map50_95 = float(np.mean(ap_per_iou))       # moyenne sur tous les IoU

        # AP par classe à IoU=0.50 (pour analyse détaillée)
        per_class = {}
        cls_names = ['pedestrian','people','bicycle','car','motorcycle',
                     'van','truck','tricycle','awning-tricycle','bus']
        for cls in range(self.num_classes):
            tp_list, conf_list, n_gt = [], [], 0
            for pred, gt in zip(self.predictions, self.ground_truths):
                gt_cls   = gt[gt[:, 4] == cls, :4] if len(gt) else torch.zeros(0,4)
                pred_cls = pred[pred[:, 5] == cls]  if len(pred) else torch.zeros(0,6)
                n_gt += len(gt_cls)
                if len(pred_cls) == 0:
                    continue
                order = pred_cls[:, 4].argsort(descending=True)
                pred_cls = pred_cls[order]
                if len(gt_cls) == 0:
                    tp_list.append(torch.zeros(len(pred_cls)))
                    conf_list.append(pred_cls[:,4])
                    continue
                iou = box_iou_xyxy(pred_cls[:,:4], gt_cls)
                matched = torch.zeros(len(gt_cls), dtype=torch.bool)
                tp = torch.zeros(len(pred_cls))
                for i in range(len(pred_cls)):
                    bv, bj = iou[i].max(0)
                    if bv >= 0.50 and not matched[bj]:
                        tp[i] = 1; matched[bj] = True
                tp_list.append(tp); conf_list.append(pred_cls[:,4])
            if tp_list and n_gt > 0:
                tp_all = torch.cat(tp_list).numpy()
                cf_all = torch.cat(conf_list).numpy()
                ord2   = cf_all.argsort()[::-1]
                tp_all = tp_all[ord2]
                ctp = tp_all.cumsum(); cfp = (1-tp_all).cumsum()
                rec = ctp / (n_gt + 1e-8); pre = ctp / (ctp + cfp + 1e-8)
                per_class[cls_names[cls]] = compute_ap(rec, pre)
            else:
                per_class[cls_names[cls]] = 0.0

        return {'mAP50': map50, 'mAP50_95': map50_95,
                'per_class_AP50': per_class}
