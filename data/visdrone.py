"""
visdrone.py
===========
Dataloader pour VisDrone-DET 2019.

Structure attendue sur disque :
    <root>/
        images/
            train/  *.jpg
            val/    *.jpg
        labels/
            train/  *.txt   (format VisDrone : x,y,w,h,score,cat,trunc,occ)
            val/    *.txt

Format annotation VisDrone (une ligne par objet) :
    x_min, y_min, width, height, score_ignore, category, truncation, occlusion

Catégories (1-indexed dans les labels, 0-indexed dans notre code) :
    0: pedestrian        5: van
    1: people            6: truck
    2: bicycle           7: tricycle
    3: car               8: awning-tricycle
    4: motorcycle        9: bus

Usage :
    from data.visdrone import build_dataloaders
    train_dl, val_dl = build_dataloaders(root='/path/to/visdrone', img_size=640)
"""

import os
import cv2
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


# Classes VisDrone valides (on ignore 0=ignored_region et 11=others)
VISDRONE_VALID_CLASSES = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
# Remapping vers 0-indexed
CLASS_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9}
NUM_CLASSES = 10


class VisDroneDataset(Dataset):
    """
    Dataset VisDrone-DET.

    Args:
        root     : dossier racine contenant images/ et labels/
        split    : 'train' ou 'val'
        img_size : taille de redimensionnement (carré)
        max_dets : nombre maximum de détections par image (pour le collate)
    """

    def __init__(self, root: str, split: str = 'train',
                 img_size: int = 640, max_dets: int = 500):
        self.img_dir = Path(root) / 'images' / split
        self.ann_dir = Path(root) / 'labels' / split
        self.img_size = img_size
        self.max_dets = max_dets

        self.samples = []
        for img_path in sorted(self.img_dir.glob('*.jpg')):
            ann_path = self.ann_dir / (img_path.stem + '.txt')
            if ann_path.exists():
                self.samples.append((img_path, ann_path))

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"Aucune image trouvée dans {self.img_dir}.\n"
                "Vérifiez la structure du dossier :\n"
                "  <root>/images/train/*.jpg\n"
                "  <root>/labels/train/*.txt"
            )

    def __len__(self):
        return len(self.samples)

    def _load_labels(self, ann_path: Path, orig_w: int, orig_h: int):
        """
        Lit le fichier annotation VisDrone et retourne les boxes en format
        [x_c, y_c, w, h, class_id] normalisé dans [0,1].
        """
        boxes = []
        with open(ann_path) as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                x, y, w, h = map(float, parts[:4])
                cat = int(parts[5])
                if cat not in VISDRONE_VALID_CLASSES:
                    continue
                if w <= 0 or h <= 0:
                    continue
                # Normalise en xywh [0,1]
                xc = (x + w / 2) / orig_w
                yc = (y + h / 2) / orig_h
                wn = w / orig_w
                hn = h / orig_h
                cls_id = CLASS_MAP[cat]
                boxes.append([xc, yc, wn, hn, cls_id])
        return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 5))

    def __getitem__(self, idx):
        img_path, ann_path = self.samples[idx]

        # ── Chargement image ─────────────────────────────────────────────────
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        # ── Annotations ──────────────────────────────────────────────────────
        boxes = self._load_labels(ann_path, orig_w, orig_h)

        # ── Redimensionnement ────────────────────────────────────────────────
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # ── Normalisation ImageNet ────────────────────────────────────────────
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (img - mean) / std

        # ── Pad boxes vers max_dets ───────────────────────────────────────────
        n = len(boxes)
        target = np.zeros((self.max_dets, 5), dtype=np.float32)
        if n > 0:
            n = min(n, self.max_dets)
            target[:n] = boxes[:n]
        mask = torch.zeros(self.max_dets, dtype=torch.bool)
        mask[:n] = True

        return {
            'image':    img,
            'boxes':    torch.from_numpy(target),   # (max_dets, 5) xywh_norm+cls
            'mask':     mask,                        # (max_dets,) bool — valide?
            'img_path': str(img_path),
            'orig_size': (orig_h, orig_w),
        }


def collate_fn(batch):
    """Collate standard — les tenseurs sont déjà paddés à max_dets."""
    images  = torch.stack([b['image']  for b in batch])
    boxes   = torch.stack([b['boxes']  for b in batch])
    masks   = torch.stack([b['mask']   for b in batch])
    paths   = [b['img_path']  for b in batch]
    sizes   = [b['orig_size'] for b in batch]
    return {'image': images, 'boxes': boxes, 'mask': masks,
            'img_path': paths, 'orig_size': sizes}


def build_dataloaders(root: str,
                      img_size: int = 640,
                      batch_size: int = 8,
                      num_workers: int = 4,
                      max_dets: int = 500):
    """
    Construit les dataloaders train et val pour VisDrone.

    Returns:
        (train_loader, val_loader)
    """
    train_ds = VisDroneDataset(root, split='train',
                               img_size=img_size, max_dets=max_dets)
    val_ds   = VisDroneDataset(root, split='val',
                               img_size=img_size, max_dets=max_dets)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, collate_fn=collate_fn,
                          pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, collate_fn=collate_fn,
                          pin_memory=True)

    print(f"[VisDrone] train={len(train_ds)} images | val={len(val_ds)} images")
    return train_dl, val_dl
