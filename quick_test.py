"""
quick_test.py
=============
Test rapide des 9 convolutions.

Deux modes :
  1. Dataset sur disque (recommandé — images visualisables) :
       python generate_synthetic.py --out_dir ./synthetic_dataset
       python quick_test.py --data ./synthetic_dataset --epochs 5 --batch 8

  2. Dataset en mémoire (fallback, zéro I/O) :
       python quick_test.py --epochs 3 --img_size 128 --batch 8
"""

import sys, os, time, argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import CONV_REGISTRY, build_detector
from utils.metrics import MAPMetric, decode_predictions


# ─────────────────────────────────────────────────────────────────────────────
# Dataset en mémoire (fallback)
# ─────────────────────────────────────────────────────────────────────────────
class InMemoryDataset(Dataset):
    """Dataset synthétique purement en mémoire — zéro I/O."""
    def __init__(self, n=200, img_size=128, num_classes=10, max_dets=20, seed=42):
        torch.manual_seed(seed)
        self.images, self.targets = [], []
        mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
        std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
        for _ in range(n):
            img = torch.rand(3, img_size, img_size) * 0.3
            n_obj = torch.randint(1, 6, (1,)).item()
            boxes = []
            for _ in range(n_obj):
                w  = torch.FloatTensor(1).uniform_(0.05, 0.25).item()
                h  = torch.FloatTensor(1).uniform_(0.05, 0.25).item()
                xc = torch.FloatTensor(1).uniform_(w/2, 1-w/2).item()
                yc = torch.FloatTensor(1).uniform_(h/2, 1-h/2).item()
                cls= torch.randint(0, num_classes, (1,)).item()
                x1,y1 = int((xc-w/2)*img_size), int((yc-h/2)*img_size)
                x2,y2 = int((xc+w/2)*img_size), int((yc+h/2)*img_size)
                c = torch.rand(3).view(3,1,1)*0.7+0.3
                img[:, y1:y2, x1:x2] = c.expand(3, y2-y1, x2-x1)
                boxes.append([xc, yc, w, h, cls])
            img = (img - mean) / std
            tgt = torch.zeros(max_dets, 5)
            n_b = min(len(boxes), max_dets)
            if n_b: tgt[:n_b] = torch.tensor(boxes[:n_b])
            mask = torch.zeros(max_dets, dtype=torch.bool)
            mask[:n_b] = True
            self.images.append(img)
            self.targets.append((tgt, mask))
    def __len__(self): return len(self.images)
    def __getitem__(self, i):
        t, m = self.targets[i]
        return self.images[i], t, m


def collate_fn(batch):
    imgs    = torch.stack([b[0] for b in batch])
    targets = torch.stack([b[1] for b in batch])
    masks   = torch.stack([b[2] for b in batch])
    return imgs, targets, masks


# Wrapper VisDrone → format (img, boxes, mask)
class VisDroneWrapper(Dataset):
    def __init__(self, ds): self.ds = ds
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        b = self.ds[i]
        return b['image'], b['boxes'], b['mask']


# ─────────────────────────────────────────────────────────────────────────────
# Perte
# ─────────────────────────────────────────────────────────────────────────────
class QuickLoss(nn.Module):
    def __init__(self, nc=10):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.sl1 = nn.SmoothL1Loss()
    def forward(self, cls_preds, reg_preds, boxes, mask):
        total = torch.tensor(0., device=cls_preds[0].device)
        for cm, rm in zip(cls_preds, reg_preds):
            B, C, H, W = cm.shape
            ct = torch.zeros_like(cm)
            rt = torch.zeros_like(rm)
            hr = torch.zeros(B,1,H,W, dtype=torch.bool, device=cm.device)
            for b in range(B):
                vb = boxes[b][mask[b]]
                if not len(vb): continue
                xi = (vb[:,0]*W).long().clamp(0,W-1)
                yi = (vb[:,1]*H).long().clamp(0,H-1)
                for i in range(len(vb)):
                    ci = vb[i,4].long()
                    ct[b,ci,yi[i],xi[i]] = 1.
                    rt[b,0,yi[i],xi[i]]  = vb[i,0]*W - xi[i]
                    rt[b,1,yi[i],xi[i]]  = vb[i,1]*H - yi[i]
                    rt[b,2,yi[i],xi[i]]  = (vb[i,2]*W).log().clamp(-4,4)
                    rt[b,3,yi[i],xi[i]]  = (vb[i,3]*H).log().clamp(-4,4)
                    hr[b,0,yi[i],xi[i]]  = True
            total = total + self.bce(cm, ct)
            if hr.any():
                total = total + 5*self.sl1(rm[hr.expand_as(rm)],
                                           rt[hr.expand_as(rt)])
        return total


# ─────────────────────────────────────────────────────────────────────────────
# Run une convolution
# ─────────────────────────────────────────────────────────────────────────────
def run_one(conv_name, train_dl, val_dl, device, epochs, img_size, nc):
    from utils.metrics import MAPMetric, decode_predictions

    model     = build_detector(conv_name, num_classes=nc, base_ch=16).to(device)
    criterion = QuickLoss(nc).to(device)
    opt       = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched     = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    metric    = MAPMetric(num_classes=nc)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    best_map50, epoch_times, train_losses = 0.0, [], []
    map_history = []   # mAP@50 par epoch

    for ep in range(1, epochs+1):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        total, t0 = 0., time.time()
        for imgs, targets, masks in train_dl:
            imgs    = imgs.to(device)
            targets = targets.to(device)
            masks   = masks.to(device)
            opt.zero_grad()
            cp, rp = model(imgs)
            loss   = criterion(cp, rp, targets, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            opt.step()
            total += loss.item()
        sched.step()
        avg     = total / len(train_dl)
        elapsed = time.time() - t0
        train_losses.append(avg)
        epoch_times.append(elapsed)

        # ── Val mAP rapide à chaque epoch ─────────────────────────────────
        model.eval()
        metric.reset()
        val_loss_ep = 0.
        with torch.no_grad():
            for imgs, targets, masks in val_dl:
                imgs    = imgs.to(device)
                targets = targets.to(device)
                masks   = masks.to(device)
                cp, rp  = model(imgs)
                val_loss_ep += criterion(cp, rp, targets, masks).item()
                preds = decode_predictions(cp, rp, img_size=img_size,
                                           conf_thresh=0.01, num_classes=nc)
                gts = []
                for b in range(len(preds)):
                    valid = targets[b][masks[b]].cpu()
                    if len(valid) == 0:
                        gts.append(torch.zeros(0, 5)); continue
                    g = torch.zeros_like(valid)
                    g[:, 0] = valid[:, 0] - valid[:, 2] / 2
                    g[:, 1] = valid[:, 1] - valid[:, 3] / 2
                    g[:, 2] = valid[:, 0] + valid[:, 2] / 2
                    g[:, 3] = valid[:, 1] + valid[:, 3] / 2
                    g[:, 4] = valid[:, 4]
                    gts.append(g)
                metric.update(preds, gts)

        val_loss_ep /= len(val_dl)
        # mAP rapide sans affichage détaillé
        map_ep = metric.compute(verbose=False)
        map50_ep = map_ep['mAP50']
        map_history.append(map50_ep)
        if map50_ep > best_map50:
            best_map50 = map50_ep

        print(f"    epoch {ep:2d}/{epochs} | "
              f"loss={avg:.4f} | "
              f"val_loss={val_loss_ep:.4f} | "
              f"mAP@50={map50_ep:.4f} | "
              f"{elapsed:.1f}s", flush=True)

    # ── Évaluation finale détaillée (par classe) ──────────────────────────
    print(f"\n  ── Évaluation finale détaillée ──", flush=True)
    model.eval()
    metric.reset()
    val_loss = 0.
    with torch.no_grad():
        for imgs, targets, masks in val_dl:
            imgs    = imgs.to(device)
            targets = targets.to(device)
            masks   = masks.to(device)
            cp, rp  = model(imgs)
            val_loss += criterion(cp, rp, targets, masks).item()
            preds = decode_predictions(cp, rp, img_size=img_size,
                                       conf_thresh=0.01, num_classes=nc)
            gts = []
            for b in range(len(preds)):
                valid = targets[b][masks[b]].cpu()
                if len(valid) == 0:
                    gts.append(torch.zeros(0, 5)); continue
                g = torch.zeros_like(valid)
                g[:, 0] = valid[:, 0] - valid[:, 2] / 2
                g[:, 1] = valid[:, 1] - valid[:, 3] / 2
                g[:, 2] = valid[:, 0] + valid[:, 2] / 2
                g[:, 3] = valid[:, 1] + valid[:, 3] / 2
                g[:, 4] = valid[:, 4]
                gts.append(g)
            metric.update(preds, gts)

    val_loss /= len(val_dl)
    # verbose=True → affiche le détail par classe avec barre visuelle
    map_res  = metric.compute(verbose=True)
    map50    = map_res['mAP50']
    map50_95 = map_res['mAP50_95']

    print(f"\n  ┌─ RÉSUMÉ {conv_name.upper()} {'─'*40}", flush=True)
    print(f"  │  val_loss   = {val_loss:.4f}", flush=True)
    print(f"  │  mAP@50     = {map50:.4f}", flush=True)
    print(f"  │  mAP@50:95  = {map50_95:.4f}", flush=True)
    print(f"  │  best mAP@50= {best_map50:.4f}", flush=True)
    print(f"  │  params     = {n_params:,}", flush=True)
    print(f"  └{'─'*50}", flush=True)

    return {
        'conv':       conv_name,
        'val_loss':   val_loss,
        'map50':      map50,
        'map50_95':   map50_95,
        'best_map50': best_map50,
        'final_loss': train_losses[-1],
        'params':     n_params,
        'avg_time_s': sum(epoch_times) / len(epoch_times),
        'total_time': sum(epoch_times),
        'map_history': map_history,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs',   type=int, default=5)
    p.add_argument('--img_size', type=int, default=128)
    p.add_argument('--batch',    type=int, default=8)
    p.add_argument('--n_train',  type=int, default=400,
                   help='Nb images en mode mémoire uniquement')
    p.add_argument('--n_val',    type=int, default=100)
    p.add_argument('--convs',    type=str, default='all',
                   help='"all" ou liste séparée par virgules')
    p.add_argument('--device',   type=str, default='')
    p.add_argument('--data',     type=str, default='',
                   help='Dossier dataset sur disque '
                        '(generate_synthetic.py ou VisDrone réel). '
                        'Si absent : mode mémoire.')
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(
        args.device if args.device
        else ('cuda' if torch.cuda.is_available() else 'cpu'))
    conv_list = list(CONV_REGISTRY) if args.convs=='all' \
                else [c.strip() for c in args.convs.split(',')]

    print('\n' + '='*65, flush=True)
    print('  SAC Quick Test', flush=True)
    print(f'  Device   : {device}', flush=True)
    print(f'  Epochs   : {args.epochs} | img_size={args.img_size} | '
          f'batch={args.batch}', flush=True)
    if args.data:
        print(f'  Dataset  : {args.data}  (fichiers PNG sur disque)', flush=True)
    else:
        print(f'  Dataset  : mémoire ({args.n_train}+{args.n_val} images)',
              flush=True)
    print('='*65, flush=True)

    nc = 10

    # ── Dataloaders ────────────────────────────────────────────────────────
    if args.data:
        from data.visdrone import VisDroneDataset, collate_fn as vc
        train_ds = VisDroneDataset(args.data, 'train', args.img_size, 50)
        val_ds   = VisDroneDataset(args.data, 'val',   args.img_size, 50)
        train_dl = DataLoader(VisDroneWrapper(train_ds), args.batch,
                              shuffle=True,  collate_fn=collate_fn, num_workers=0)
        val_dl   = DataLoader(VisDroneWrapper(val_ds),   args.batch,
                              shuffle=False, collate_fn=collate_fn, num_workers=0)
        print(f'  ✓ train={len(train_ds)} | val={len(val_ds)} images chargées',
              flush=True)
    else:
        train_ds = InMemoryDataset(args.n_train, args.img_size, nc, seed=42)
        val_ds   = InMemoryDataset(args.n_val,   args.img_size, nc, seed=99)
        train_dl = DataLoader(train_ds, args.batch, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
        val_dl   = DataLoader(val_ds,   args.batch, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    # ── Benchmark ─────────────────────────────────────────────────────────
    results, t_all = [], time.time()

    for i, conv in enumerate(conv_list):
        print(f'\n{"─"*65}', flush=True)
        print(f'  [{i+1}/{len(conv_list)}] {conv.upper()}', flush=True)
        print(f'{"─"*65}', flush=True)
        try:
            res = run_one(conv, train_dl, val_dl, device,
                          args.epochs, args.img_size, nc)
            results.append(res)
            print(f'  ✓ val_loss={res["val_loss"]:.4f} | '
                  f'params={res["params"]:,} | '
                  f'{res["avg_time_s"]:.1f}s/epoch', flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'  ✗ ERREUR : {e}', flush=True)
            results.append({'conv':conv,'val_loss':float('nan'),
                            'best_loss':float('nan'),'final_loss':float('nan'),
                            'params':0,'avg_time_s':0,'total_time':0})

    # ── Tableau ────────────────────────────────────────────────────────────
    LABELS = {
        'standard':       'Standard Conv',
        'deformable':     'Deformable Conv',
        'dynamic_filter': 'Dynamic Filter Net',
        'dynamic_conv':   'Dynamic Conv',
        'condconv':       'CondConv',
        'pac':            'PAC',
        'knconv':         'KNConv',
        'hyperconv':      'HyperConv',
        'sac':            'SAC',
        'sac_fast':       'SAC Fast',
        'pwc':            'PWC (ours)',
    }

    # Trie par mAP@50 décroissant (métrique principale)
    valid = sorted(
        [r for r in results if r.get('map50', float('nan')) == r.get('map50', float('nan'))],
        key=lambda r: r.get('map50', 0), reverse=True
    )

    sep = '='*80
    print(f'\n{sep}', flush=True)
    print(f"  RÉSULTATS — {args.epochs} epochs | img_size={args.img_size} | "
          f"dataset={'disque' if args.data else 'mémoire'}", flush=True)
    print(f"  Trié par mAP@50 décroissant", flush=True)
    print(sep, flush=True)
    header = (f"  {'Rang':<5} {'Méthode':<22} {'mAP@50':>8} "
              f"{'mAP@50:95':>10} {'Val loss':>9} {'Params':>10} {'s/ep':>6}")
    print(header, flush=True)
    print('-'*80, flush=True)

    for rank, r in enumerate(results, 1):
        label   = LABELS.get(r['conv'], r['conv'])
        star    = ' ★' if r['conv'] in ('pwc', 'sac') else ''
        is_best = valid and r['conv'] == valid[0]['conv']
        mark    = '→ ' if is_best else '  '

        map50    = r.get('map50',    float('nan'))
        map5095  = r.get('map50_95', float('nan'))
        vl       = r.get('val_loss', float('nan'))

        m50_s   = f"{map50:.4f}"   if map50   == map50   else 'ERR'
        m5095_s = f"{map5095:.4f}" if map5095 == map5095 else 'ERR'
        vl_s    = f"{vl:.4f}"      if vl      == vl      else 'ERR'

        print(f"  {mark}{rank:<4} {label+star:<22} {m50_s:>8} "
              f"{m5095_s:>10} {vl_s:>9} "
              f"{r['params']:>10,} {r['avg_time_s']:>6.1f}", flush=True)

    print(sep, flush=True)
    print(f'  Temps total : {(time.time()-t_all)/60:.1f} min', flush=True)
    if valid:
        best = valid[0]
        print(f'  → Meilleur mAP@50 : {best["conv"].upper()} '
              f'({best.get("map50",0):.4f})', flush=True)

    # ── CSV ────────────────────────────────────────────────────────────────
    import csv
    os.makedirs('./runs', exist_ok=True)
    path = './runs/quick_test_results.csv'
    fieldnames = ['conv', 'map50', 'map50_95', 'val_loss',
                  'final_loss', 'params', 'avg_time_s', 'total_time']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)
    print(f'  Résultats → {path}', flush=True)


if __name__ == '__main__':
    main()
