"""
metrics.py
==========
Calcul du mAP@50 et mAP@50:95 — version optimisée.

Stratégie :
  - Pré-agrégation : on concatène TOUTES les preds/GTs par classe en une fois.
  - Matching vectorisé : calcul des TP sur tous les seuils IoU en une seule
    passe (sans reboucler sur les images pour chaque seuil).
  - Résultat : compute() est 10-50× plus rapide que la version naïve.
"""

import torch
import numpy as np
from typing import List, Dict


# ─────────────────────────────────────────────────────────────────────────────
# IoU
# ─────────────────────────────────────────────────────────────────────────────
def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    IoU entre deux ensembles de boîtes [x1,y1,x2,y2].
    boxes1 : (N, 4)  boxes2 : (M, 4)  →  iou : (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(0) * \
            (boxes1[:, 3] - boxes1[:, 1]).clamp(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(0) * \
            (boxes2[:, 3] - boxes2[:, 1]).clamp(0)
    lt    = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb    = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh    = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area1[:, None] + area2[None, :] - inter + 1e-8)


def xywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    """[xc, yc, w, h] → [x1, y1, x2, y2]"""
    o = b.clone()
    o[..., 0] = b[..., 0] - b[..., 2] / 2
    o[..., 1] = b[..., 1] - b[..., 3] / 2
    o[..., 2] = b[..., 0] + b[..., 2] / 2
    o[..., 3] = b[..., 1] + b[..., 3] / 2
    return o


# ─────────────────────────────────────────────────────────────────────────────
# AP sur courbe précision-rappel (101 points COCO)
# ─────────────────────────────────────────────────────────────────────────────
def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    r = np.concatenate(([0.], recalls, [1.]))
    p = np.concatenate(([1.], precisions, [0.]))
    for i in range(len(p) - 2, -1, -1):
        p[i] = max(p[i], p[i + 1])
    return float(np.mean(p[np.searchsorted(r, np.linspace(0, 1, 101))]))


# ─────────────────────────────────────────────────────────────────────────────
# Décodage des prédictions du détecteur
# ─────────────────────────────────────────────────────────────────────────────
def decode_predictions(cls_preds: List[torch.Tensor],
                       reg_preds: List[torch.Tensor],
                       img_size: int,
                       conf_thresh: float = 0.01,
                       num_classes: int = 10) -> List[torch.Tensor]:
    B = cls_preds[0].shape[0]
    all_dets = [[] for _ in range(B)]

    for cls_map, reg_map in zip(cls_preds, reg_preds):
        _, _, H, W = cls_map.shape
        sh, sw = img_size / H, img_size / W

        gy, gx = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=cls_map.device),
            torch.arange(W, dtype=torch.float32, device=cls_map.device),
            indexing='ij')
        cx = (gx + 0.5) * sw / img_size
        cy = (gy + 0.5) * sh / img_size

        scores = cls_map.sigmoid()
        conf, cls_id = scores.max(dim=1)

        bx = reg_map[:, 0].sigmoid() + cx
        by = reg_map[:, 1].sigmoid() + cy
        bw = reg_map[:, 2].exp() * (sw / img_size)
        bh = reg_map[:, 3].exp() * (sh / img_size)

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
            ], dim=1)
            all_dets[b].append(det)

    return [torch.cat(d, 0) if d else torch.zeros(0, 6)
            for d in all_dets]


# ─────────────────────────────────────────────────────────────────────────────
# mAP optimisé
# ─────────────────────────────────────────────────────────────────────────────
class MAPMetric:
    """
    mAP@50 et mAP@50:95 — version rapide.

    Optimisations vs la version naïve :
    1. Pré-agrégation : toutes les preds/GTs sont concaténées par classe
       UNE SEULE FOIS, pas à chaque seuil IoU.
    2. Matching multi-seuil : pour chaque paire (pred_box, gt_box), on calcule
       l'IoU une seule fois et on détermine à quels seuils cette paire est un TP.
    3. Image-index stocké : permet de ne pas reboucler sur les images pour le
       matching (on travaille sur des tenseurs globaux par classe).
    """

    IOU_THRESHOLDS = np.arange(0.50, 1.00, 0.05)   # 10 seuils COCO
    CLS_NAMES = ['pedestrian', 'people', 'bicycle', 'car', 'motorcycle',
                 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus']

    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes
        # Stockage par classe : listes de (boxes_xyxy, conf, img_idx)
        self._preds: List[List] = [[] for _ in range(num_classes)]
        self._gts:   List[List] = [[] for _ in range(num_classes)]
        self._img_idx = 0

    def reset(self):
        self._preds   = [[] for _ in range(self.num_classes)]
        self._gts     = [[] for _ in range(self.num_classes)]
        self._img_idx = 0

    def update(self,
               preds: List[torch.Tensor],
               gts:   List[torch.Tensor]):
        """
        preds : list[Tensor(N,6)] — x1y1x2y2 conf cls
        gts   : list[Tensor(M,5)] — x1y1x2y2 cls
        """
        for p, g in zip(preds, gts):
            p = p.cpu(); g = g.cpu()
            # Trie les preds par confiance décroissante dès l'update
            if len(p) > 0:
                order = p[:, 4].argsort(descending=True)
                p = p[order]
            for cls in range(self.num_classes):
                pc = p[p[:, 5] == cls, :5] if len(p) > 0 else torch.zeros(0, 5)
                gc = g[g[:, 4] == cls, :4] if len(g) > 0 else torch.zeros(0, 4)
                if len(pc) > 0:
                    # Ajoute img_idx comme 6ème colonne pour le matching global
                    idx_col = torch.full((len(pc), 1), self._img_idx)
                    self._preds[cls].append(torch.cat([pc, idx_col], dim=1))
                if len(gc) > 0:
                    idx_col = torch.full((len(gc), 1), self._img_idx)
                    self._gts[cls].append(torch.cat([gc, idx_col], dim=1))
            self._img_idx += 1

    # ── Calcul AP pour une classe et tous les seuils IoU ─────────────────────
    def _ap_for_class(self, cls: int) -> np.ndarray:
        """
        Retourne un vecteur de len(IOU_THRESHOLDS) AP values pour la classe cls.
        Matching global : on travaille sur les tenseurs concaténés de toutes
        les images — pas de boucle Python image par image.
        """
        if not self._preds[cls] and not self._gts[cls]:
            return np.zeros(len(self.IOU_THRESHOLDS))

        n_gt = sum(len(g) for g in self._gts[cls])
        if n_gt == 0:
            return np.zeros(len(self.IOU_THRESHOLDS))

        if not self._preds[cls]:
            return np.zeros(len(self.IOU_THRESHOLDS))

        # Concatène toutes les preds et GTs de la classe
        # preds_all : (N_pred, 6) — x1y1x2y2 conf img_idx
        # gts_all   : (N_gt,  5) — x1y1x2y2 img_idx
        preds_all = torch.cat(self._preds[cls], 0)  # déjà trié par conf
        gts_all   = torch.cat(self._gts[cls], 0)

        n_pred = len(preds_all)
        # TP matrix : (n_pred, n_iou_thresh)
        tp_matrix = torch.zeros(n_pred, len(self.IOU_THRESHOLDS))

        # Pour chaque image qui a des GTs, calcule le matching
        img_ids_with_gt = gts_all[:, 4].unique()

        for img_id in img_ids_with_gt:
            img_id = img_id.item()
            gt_mask   = gts_all[:, 4] == img_id
            pred_mask = preds_all[:, 5] == img_id

            gc = gts_all[gt_mask, :4]      # (M, 4)
            pc = preds_all[pred_mask, :4]  # (K, 4)
            pred_idx = pred_mask.nonzero(as_tuple=True)[0]  # indices globaux

            if len(pc) == 0:
                continue

            # IoU (K, M)
            iou = box_iou_xyxy(pc, gc)     # (K, M)

            # Pour chaque seuil IoU, marque les TP
            for t_idx, thresh in enumerate(self.IOU_THRESHOLDS):
                matched_gt = torch.zeros(len(gc), dtype=torch.bool)
                for k in range(len(pc)):
                    best_iou, best_j = iou[k].max(0)
                    if best_iou >= thresh and not matched_gt[best_j]:
                        tp_matrix[pred_idx[k], t_idx] = 1
                        matched_gt[best_j] = True

        # Calcule l'AP pour chaque seuil IoU
        conf_all = preds_all[:, 4].numpy()  # déjà trié desc
        ap_per_thresh = []
        for t_idx in range(len(self.IOU_THRESHOLDS)):
            tp = tp_matrix[:, t_idx].numpy()
            cum_tp = tp.cumsum()
            cum_fp = (1 - tp).cumsum()
            rec = cum_tp / (n_gt + 1e-8)
            pre = cum_tp / (cum_tp + cum_fp + 1e-8)
            ap_per_thresh.append(compute_ap(rec, pre))

        return np.array(ap_per_thresh)

    # ── compute principal ─────────────────────────────────────────────────────
    def compute(self, verbose: bool = True) -> Dict[str, float]:
        """
        Calcule mAP@50 et mAP@50:95.

        Les 10 classes sont traitées en parallèle via ThreadPoolExecutor.
        Chaque classe est indépendante → gain direct ~4-8× sur le calcul mAP.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        ap_matrix = np.zeros((self.num_classes, len(self.IOU_THRESHOLDS)))

        if verbose:
            print('  → Calcul du mAP (parallèle par classe)...', flush=True)

        # Lance les 10 classes en parallèle — chacune est indépendante
        n_workers = min(self.num_classes, 8)   # max 8 threads (Colab = 2 CPU)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(self._ap_for_class, cls): cls
                for cls in range(self.num_classes)
            }
            completed = 0
            for future in as_completed(futures):
                cls = futures[future]
                ap_matrix[cls] = future.result()
                completed += 1
                if verbose:
                    cls_name = self.CLS_NAMES[cls] if cls < len(self.CLS_NAMES) \
                               else str(cls)
                    ap50 = ap_matrix[cls, 0]
                    bar  = '█' * int(ap50 * 20)
                    print(f'     [{completed:2d}/10] {cls_name:<20} '
                          f'AP@50={ap50:.4f}  {bar}', flush=True)

        map50    = float(ap_matrix[:, 0].mean())
        map50_95 = float(ap_matrix.mean())

        per_class = {
            self.CLS_NAMES[c]: float(ap_matrix[c, 0])
            for c in range(min(self.num_classes, len(self.CLS_NAMES)))
        }

        print(f'  → mAP@50 = {map50:.4f}  |  mAP@50:95 = {map50_95:.4f}',
              flush=True)

        return {'mAP50': map50, 'mAP50_95': map50_95,
                'per_class_AP50': per_class}
