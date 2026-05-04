"""
quick_test.py
=============
Test rapide des 9 convolutions sur un dataset synthétique généré en mémoire.
Aucune donnée réelle nécessaire — tourne en ~5-10 min sur Colab GPU.

Usage :
    python quick_test.py
    python quick_test.py --epochs 3 --img_size 128 --batch 8
"""

import sys
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Ajoute le dossier courant au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import CONV_REGISTRY, build_detector

# ─────────────────────────────────────────────────────────────────────────────
# Dataset synthétique
# ─────────────────────────────────────────────────────────────────────────────
class SyntheticDetectionDataset(Dataset):
    """
    Génère des images synthétiques avec des objets géométriques simples
    (rectangles colorés sur fond aléatoire) — mimique les conditions de
    détection de petits objets de VisDrone.

    Chaque image contient 1-5 objets de taille variable (5-30% de l'image).
    Tout est généré en mémoire avec torch — zéro I/O disque.
    """

    def __init__(self, n_samples=200, img_size=128,
                 num_classes=10, max_dets=20, seed=42):
        self.n       = n_samples
        self.size    = img_size
        self.nc      = num_classes
        self.max_det = max_dets
        torch.manual_seed(seed)

        # Pré-génère tout en mémoire
        print(f"  Génération de {n_samples} images synthétiques "
              f"({img_size}×{img_size})...", flush=True)
        self.images  = []
        self.targets = []

        for _ in range(n_samples):
            img = torch.rand(3, img_size, img_size) * 0.3  # fond sombre
            n_obj = torch.randint(1, 6, (1,)).item()
            boxes = []

            for _ in range(n_obj):
                # Taille de l'objet : 5-30% de l'image (petits objets)
                w = torch.FloatTensor(1).uniform_(0.05, 0.30).item()
                h = torch.FloatTensor(1).uniform_(0.05, 0.30).item()
                xc = torch.FloatTensor(1).uniform_(w/2, 1 - w/2).item()
                yc = torch.FloatTensor(1).uniform_(h/2, 1 - h/2).item()
                cls_id = torch.randint(0, num_classes, (1,)).item()

                # Dessine l'objet (rectangle lumineux)
                x1 = int((xc - w/2) * img_size)
                y1 = int((yc - h/2) * img_size)
                x2 = int((xc + w/2) * img_size)
                y2 = int((yc + h/2) * img_size)
                color = torch.rand(3).view(3, 1, 1) * 0.7 + 0.3
                img[:, y1:y2, x1:x2] = color.expand(3, y2-y1, x2-x1)

                boxes.append([xc, yc, w, h, cls_id])

            # Normalisation ImageNet
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
            img  = (img - mean) / std

            # Pad jusqu'à max_dets
            target = torch.zeros(self.max_det, 5)
            n = min(len(boxes), self.max_det)
            if n > 0:
                target[:n] = torch.tensor(boxes[:n])

            mask = torch.zeros(self.max_det, dtype=torch.bool)
            mask[:n] = True

            self.images.append(img)
            self.targets.append((target, mask))

        print(f"  ✓ Dataset prêt — {n_samples} images en mémoire", flush=True)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        target, mask = self.targets[idx]
        return self.images[idx], target, mask


def collate_fn(batch):
    imgs    = torch.stack([b[0] for b in batch])
    targets = torch.stack([b[1] for b in batch])
    masks   = torch.stack([b[2] for b in batch])
    return imgs, targets, masks


# ─────────────────────────────────────────────────────────────────────────────
# Perte simplifiée
# ─────────────────────────────────────────────────────────────────────────────
class QuickLoss(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.nc  = num_classes
        self.bce = nn.BCEWithLogitsLoss()
        self.sl1 = nn.SmoothL1Loss()

    def forward(self, cls_preds, reg_preds, boxes, mask):
        total = torch.tensor(0., device=cls_preds[0].device)
        for cls_map, reg_map in zip(cls_preds, reg_preds):
            B, C, H, W = cls_map.shape
            cls_tgt = torch.zeros_like(cls_map)
            reg_tgt = torch.zeros_like(reg_map)
            has_reg = torch.zeros(B,1,H,W, dtype=torch.bool,
                                  device=cls_map.device)
            for b in range(B):
                vb = boxes[b][mask[b]]
                if len(vb) == 0:
                    continue
                xi = (vb[:,0]*W).long().clamp(0,W-1)
                yi = (vb[:,1]*H).long().clamp(0,H-1)
                for i in range(len(vb)):
                    ci = vb[i,4].long()
                    cls_tgt[b,ci,yi[i],xi[i]] = 1.
                    reg_tgt[b,0,yi[i],xi[i]]  = vb[i,0]*W - xi[i]
                    reg_tgt[b,1,yi[i],xi[i]]  = vb[i,1]*H - yi[i]
                    reg_tgt[b,2,yi[i],xi[i]]  = (vb[i,2]*W).log().clamp(-4,4)
                    reg_tgt[b,3,yi[i],xi[i]]  = (vb[i,3]*H).log().clamp(-4,4)
                    has_reg[b,0,yi[i],xi[i]]   = True
            total = total + self.bce(cls_map, cls_tgt)
            if has_reg.any():
                total = total + 5 * self.sl1(
                    reg_map[has_reg.expand_as(reg_map)],
                    reg_tgt[has_reg.expand_as(reg_tgt)]
                )
        return total


# ─────────────────────────────────────────────────────────────────────────────
# Entraînement rapide d'une conv
# ─────────────────────────────────────────────────────────────────────────────
def run_one(conv_name, train_dl, val_dl, device,
            epochs, img_size, num_classes):

    model     = build_detector(conv_name, num_classes=num_classes,
                               base_ch=16).to(device)
    criterion = QuickLoss(num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs, eta_min=1e-5)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    best_loss = float('inf')
    epoch_times = []
    train_losses = []

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.
        t0 = time.time()
        for imgs, targets, masks in train_dl:
            imgs    = imgs.to(device)
            targets = targets.to(device)
            masks   = masks.to(device)
            optimizer.zero_grad()
            cp, rp  = model(imgs)
            loss    = criterion(cp, rp, targets, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        avg_loss = total_loss / len(train_dl)
        train_losses.append(avg_loss)
        elapsed  = time.time() - t0
        epoch_times.append(elapsed)

        if avg_loss < best_loss:
            best_loss = avg_loss

        print(f"    epoch {epoch:2d}/{epochs} | "
              f"loss={avg_loss:.4f} | {elapsed:.1f}s", flush=True)

    # ── Val loss ──────────────────────────────────────────────────────────
    model.eval()
    val_loss = 0.
    with torch.no_grad():
        for imgs, targets, masks in val_dl:
            imgs    = imgs.to(device)
            targets = targets.to(device)
            masks   = masks.to(device)
            cp, rp  = model(imgs)
            val_loss += criterion(cp, rp, targets, masks).item()
    val_loss /= len(val_dl)

    return {
        'conv':       conv_name,
        'val_loss':   val_loss,
        'best_loss':  best_loss,
        'final_loss': train_losses[-1],
        'params':     n_params,
        'avg_time_s': sum(epoch_times) / len(epoch_times),
        'total_time': sum(epoch_times),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description='Test rapide des 9 convolutions sur données synthétiques')
    p.add_argument('--epochs',   type=int,   default=5)
    p.add_argument('--img_size', type=int,   default=128)
    p.add_argument('--batch',    type=int,   default=8)
    p.add_argument('--n_train',  type=int,   default=400,
                   help='Nombre d\'images train synthétiques')
    p.add_argument('--n_val',    type=int,   default=100,
                   help='Nombre d\'images val synthétiques')
    p.add_argument('--convs',    type=str,   default='all',
                   help='Convolutions à tester, séparées par virgule, '
                        'ou "all" pour toutes les 9')
    p.add_argument('--device',   type=str,   default='')
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        args.device if args.device
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )

    # Sélection des convolutions à tester
    if args.convs == 'all':
        conv_list = list(CONV_REGISTRY.keys())
    else:
        conv_list = [c.strip() for c in args.convs.split(',')]

    print('\n' + '='*65, flush=True)
    print('  SAC Quick Test — Dataset Synthétique', flush=True)
    print(f'  Device   : {device}', flush=True)
    print(f'  Epochs   : {args.epochs}', flush=True)
    print(f'  img_size : {args.img_size}', flush=True)
    print(f'  batch    : {args.batch}', flush=True)
    print(f'  n_train  : {args.n_train}  |  n_val : {args.n_val}', flush=True)
    print(f'  Convs    : {conv_list}', flush=True)
    print('='*65, flush=True)

    # ── Dataset synthétique ───────────────────────────────────────────────
    num_classes = 10
    train_ds = SyntheticDetectionDataset(
        args.n_train, args.img_size, num_classes, seed=42)
    val_ds   = SyntheticDetectionDataset(
        args.n_val,   args.img_size, num_classes, seed=99)

    train_dl = DataLoader(train_ds, batch_size=args.batch,
                          shuffle=True,  collate_fn=collate_fn,
                          num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch,
                          shuffle=False, collate_fn=collate_fn,
                          num_workers=0)

    # ── Benchmark des convolutions ────────────────────────────────────────
    results = []
    total_start = time.time()

    for i, conv_name in enumerate(conv_list):
        sep = '─' * 65
        print(f'\n{sep}', flush=True)
        print(f'  [{i+1}/{len(conv_list)}] Convolution : '
              f'{conv_name.upper()}', flush=True)
        print(sep, flush=True)

        try:
            res = run_one(
                conv_name, train_dl, val_dl, device,
                args.epochs, args.img_size, num_classes
            )
            results.append(res)
            print(f'  ✓ val_loss={res["val_loss"]:.4f} | '
                  f'params={res["params"]:,} | '
                  f'{res["avg_time_s"]:.1f}s/epoch', flush=True)
        except Exception as e:
            print(f'  ✗ ERREUR : {e}', flush=True)
            results.append({'conv': conv_name, 'val_loss': float('nan'),
                            'best_loss': float('nan'),
                            'final_loss': float('nan'),
                            'params': 0,
                            'avg_time_s': 0, 'total_time': 0})

    total_elapsed = time.time() - total_start

    # ── Tableau comparatif ────────────────────────────────────────────────
    print('\n' + '='*75, flush=True)
    print(f'  RÉSULTATS — {args.epochs} epochs | '
          f'img_size={args.img_size} | dataset synthétique', flush=True)
    print('='*75, flush=True)

    # Trie par val_loss croissant
    valid_res = [r for r in results if r['val_loss'] == r['val_loss']]
    valid_res.sort(key=lambda r: r['val_loss'])

    LABELS = {
        'standard':       'Standard Conv',
        'deformable':     'Deformable Conv',
        'dynamic_filter': 'Dynamic Filter Net',
        'dynamic_conv':   'Dynamic Conv',
        'condconv':       'CondConv',
        'pac':            'PAC',
        'knconv':         'KNConv',
        'hyperconv':      'HyperConv',
        'sac':            'SAC (ours)',
    }

    header = f"{'Rang':<5} {'Méthode':<25} {'Val loss':>9} " \
             f"{'Best loss':>10} {'Params':>10} {'s/epoch':>8}"
    print(header, flush=True)
    print('-'*75, flush=True)

    for rank, r in enumerate(results, 1):
        label  = LABELS.get(r['conv'], r['conv'])
        star   = ' ★' if r['conv'] == 'sac' else ''
        best   = rank == 1

        marker = '→ ' if best else '  '
        print(f"{marker}{rank:<4} {label + star:<25} "
              f"{r['val_loss']:>9.4f} "
              f"{r['best_loss']:>10.4f} "
              f"{r['params']:>10,} "
              f"{r['avg_time_s']:>8.1f}",
              flush=True)

    print('='*75, flush=True)
    print(f'  Temps total : {total_elapsed/60:.1f} min', flush=True)
    print(f'  → Meilleure val_loss : '
          f'{valid_res[0]["conv"].upper() if valid_res else "N/A"}',
          flush=True)

    # Sauvegarde CSV
    import csv
    os.makedirs('./runs', exist_ok=True)
    csv_path = './runs/quick_test_results.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'conv','val_loss','best_loss','final_loss',
            'params','avg_time_s','total_time'])
        w.writeheader()
        w.writerows(results)
    print(f'\n  Résultats sauvegardés → {csv_path}', flush=True)


if __name__ == '__main__':
    main()
