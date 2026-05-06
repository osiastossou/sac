"""
evaluate.py
===========
Charge les meilleurs checkpoints de chaque convolution et produit :
  1. Un tableau comparatif complet (console + CSV)
  2. Un fichier results_summary.csv

Usage :
    python evaluate.py --data /path/to/visdrone --out_dir ./runs
"""

import os
import sys
import csv
import argparse
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from models import build_detector, CONV_REGISTRY
from data.visdrone import build_dataloaders, NUM_CLASSES
from utils import MAPMetric, decode_predictions
from train import validate


CONV_LABELS = {
    "standard":       "Standard Conv",
    "deformable":     "Deformable Conv (Dai 2017)",
    "dynamic_filter": "Dynamic Filter Net (De Brabandere 2016)",
    "dynamic_conv":   "Dynamic Conv (Chen 2020)",
    "condconv":       "CondConv (Yang 2019)",
    "pac":            "PAC (Su 2019)",
    "knconv":         "KNConv (Nasirigerdeh 2024)",
    "hyperconv":      "HyperConv (Ha 2017)",
    "sac":            "SAC (ours)",
}


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data',     type=str, required=True)
    p.add_argument('--out_dir',  type=str, default='./runs')
    p.add_argument('--batch',    type=int, default=8)
    p.add_argument('--img_size', type=int, default=640)
    p.add_argument('--workers',  type=int, default=2)
    p.add_argument('--base_ch',  type=int, default=32)
    p.add_argument('--device',   type=str, default='')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        args.device if args.device
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )

    _, val_dl = build_dataloaders(
        args.data, img_size=args.img_size,
        batch_size=args.batch, num_workers=args.workers
    )
    metric = MAPMetric(num_classes=NUM_CLASSES)

    results = []

    for conv_name, label in CONV_LABELS.items():
        ckpt_path = Path(args.out_dir) / conv_name / 'best.pt'
        if not ckpt_path.exists():
            print(f"[SKIP] {label} — checkpoint introuvable : {ckpt_path}")
            results.append({
                'conv': conv_name, 'label': label,
                'mAP50': 'N/A', 'mAP50_95': 'N/A', 'params': 'N/A'
            })
            continue

        print(f"\n[EVAL] {label}")
        ckpt = torch.load(ckpt_path, map_location=device)

        model = build_detector(conv_name, num_classes=NUM_CLASSES,
                               base_ch=args.base_ch).to(device)
        model.load_state_dict(ckpt['state_dict'])

        val_res = validate(model, val_dl, metric, device, args.img_size)
        n_params = count_params(model)

        print(f"  mAP@50      = {val_res['mAP50']:.4f}")
        print(f"  mAP@50:95   = {val_res['mAP50_95']:.4f}")
        print(f"  Paramètres  = {n_params:,}")

        results.append({
            'conv':     conv_name,
            'label':    label,
            'mAP50':    f"{val_res['mAP50']:.4f}",
            'mAP50_95': f"{val_res['mAP50_95']:.4f}",
            'params':   f"{n_params:,}",
        })

    # ── Tableau console ───────────────────────────────────────────────────────
    w_label = 42
    print("\n" + "=" * 80)
    print(f"{'Méthode':<{w_label}} {'mAP@50':>8} {'mAP@50:95':>10} {'Params':>12}")
    print("-" * 80)
    for r in results:
        star = " ★" if r['conv'] == 'sac' else ""
        print(f"{r['label'] + star:<{w_label}} "
              f"{r['mAP50']:>8} {r['mAP50_95']:>10} {r['params']:>12}")
    print("=" * 80)

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = Path(args.out_dir) / 'results_summary.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['conv','label','mAP50',
                                               'mAP50_95','params'])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nRésultats sauvegardés dans : {csv_path}")


if __name__ == '__main__':
    main()
