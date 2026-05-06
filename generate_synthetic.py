"""
generate_synthetic.py
=====================
Génère un dataset synthétique de détection d'objets et le sauvegarde sur disque
au format VisDrone (compatible avec le dataloader existant).

Les images sont de vraies PNG visualisables — objets géométriques colorés
(rectangles, cercles, triangles) sur fonds variés, simulant les conditions
de détection de petits objets de VisDrone.

Structure produite :
    <out_dir>/
        images/
            train/  0000.png ... N-1.png
            val/    0000.png ... M-1.png
        annotations/
            train/  0000.txt ... N-1.txt
            val/    0000.txt ... M-1.txt
        preview/              ← grille de 16 images pour visualisation rapide
            preview_train.png
            preview_val.png
        dataset_info.json     ← métadonnées

Format annotation (compatible VisDrone) :
    x_min, y_min, width, height, score, category, truncation, occlusion

Usage :
    python generate_synthetic.py
    python generate_synthetic.py --out_dir /content/synthetic --n_train 600 --n_val 150
"""

import os
import json
import argparse
import random
import math
import numpy as np
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_OK = True
except ImportError:
    PIL_OK = False

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# Palettes et classes
# ─────────────────────────────────────────────────────────────────────────────

# Classes VisDrone (cat id 1-indexed comme dans les annotations VisDrone)
CLASSES = {
    1: 'pedestrian',
    2: 'people',
    3: 'bicycle',
    4: 'car',
    5: 'motorcycle',
    6: 'van',
    7: 'truck',
    8: 'tricycle',
    9: 'awning-tricycle',
    10: 'bus',
}

# Couleur distinctive par classe (R, G, B)
CLASS_COLORS = {
    1:  (255, 80,  80),   # pedestrian — rouge vif
    2:  (255, 160, 80),   # people     — orange
    3:  (80,  200, 80),   # bicycle    — vert
    4:  (80,  120, 255),  # car        — bleu
    5:  (200, 80,  200),  # motorcycle — violet
    6:  (80,  220, 220),  # van        — cyan
    7:  (220, 180, 80),   # truck      — jaune-or
    8:  (180, 100, 60),   # tricycle   — marron
    9:  (100, 180, 100),  # awning     — vert clair
    10: (200, 200, 80),   # bus        — jaune
}

# Taille typique de chaque classe (fraction de l'image) — min, max
CLASS_SIZES = {
    1:  (0.02, 0.06),   # piéton — très petit
    2:  (0.03, 0.08),   # personne
    3:  (0.03, 0.08),   # vélo
    4:  (0.05, 0.15),   # voiture — moyen
    5:  (0.03, 0.07),   # moto
    6:  (0.07, 0.18),   # van — plus grand
    7:  (0.08, 0.22),   # camion
    8:  (0.03, 0.08),   # tricycle
    9:  (0.04, 0.10),   # awning-tricycle
    10: (0.10, 0.25),   # bus — grand
}

# Rapport largeur/hauteur typique par classe
CLASS_ASPECT = {
    1:  (0.3, 0.6),   # piéton : étroit et haut
    2:  (0.4, 0.7),
    3:  (0.8, 1.5),   # vélo : plus large
    4:  (1.5, 2.5),   # voiture : largeur > hauteur
    5:  (0.8, 1.4),
    6:  (1.8, 2.8),
    7:  (2.0, 3.5),
    8:  (1.0, 1.8),
    9:  (1.2, 2.0),
    10: (2.5, 4.0),   # bus : très large
}


# ─────────────────────────────────────────────────────────────────────────────
# Génération d'un fond d'image
# ─────────────────────────────────────────────────────────────────────────────
def make_background(img_size: int, rng: random.Random) -> np.ndarray:
    """
    Génère un fond réaliste : gradient + bruit + texture optionnelle.
    """
    H = W = img_size
    bg_type = rng.choice(['gradient', 'noise', 'grid', 'urban'])

    if bg_type == 'gradient':
        # Gradient 2D — simule une vue aérienne
        c1 = np.array([rng.randint(30, 100)] * 3, dtype=np.float32)
        c2 = np.array([rng.randint(80, 180)] * 3, dtype=np.float32)
        y_grad = np.linspace(0, 1, H).reshape(-1, 1)
        img = (c1 * (1 - y_grad) + c2 * y_grad).astype(np.uint8)
        img = np.broadcast_to(img, (H, W, 3)).copy()

    elif bg_type == 'noise':
        # Fond bruité sombre — simule asphalte/toit
        base = rng.randint(40, 100)
        img  = (np.random.RandomState(rng.randint(0, 99999))
                .randint(base - 20, base + 20, (H, W, 3))
                .astype(np.uint8))

    elif bg_type == 'grid':
        # Grille — simule carrelage/parking vu du dessus
        img = np.ones((H, W, 3), dtype=np.uint8) * rng.randint(60, 120)
        step = rng.randint(20, 50)
        color_line = rng.randint(80, 160)
        for i in range(0, H, step):
            img[i, :] = color_line
        for j in range(0, W, step):
            img[:, j] = color_line

    else:  # urban
        # Fond multi-zones — simule routes + trottoirs
        img = np.ones((H, W, 3), dtype=np.uint8) * rng.randint(50, 90)
        # Quelques bandes horizontales
        n_bands = rng.randint(2, 5)
        for _ in range(n_bands):
            y0 = rng.randint(0, H)
            h  = rng.randint(H // 10, H // 3)
            c  = rng.randint(70, 140)
            img[y0:y0+h, :] = c

    # Ajoute un léger bruit sur tous les types
    noise = np.random.RandomState(rng.randint(0, 99999)).randint(
        -10, 10, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Dessin d'un objet
# ─────────────────────────────────────────────────────────────────────────────
def draw_object(img: np.ndarray, cls_id: int,
                x1: int, y1: int, x2: int, y2: int,
                rng: random.Random) -> None:
    """
    Dessine un objet synthétique pour la classe cls_id dans la boîte [x1,y1,x2,y2].
    Chaque classe a une forme et une couleur distinctives.
    """
    color = CLASS_COLORS[cls_id]
    # Variation légère de couleur pour éviter un look trop uniforme
    color = tuple(min(255, max(0, c + rng.randint(-30, 30))) for c in color)
    w = x2 - x1
    h = y2 - y1

    if cls_id in (1, 2):
        # Piéton / personne : rectangle vertical avec "tête" (cercle)
        body_h = int(h * 0.65)
        head_r = max(2, int(w * 0.35))
        # Corps
        cv2.rectangle(img, (x1, y1 + head_r * 2), (x2, y1 + head_r * 2 + body_h),
                      color, -1)
        # Tête
        cx = (x1 + x2) // 2
        cy = y1 + head_r
        cv2.circle(img, (cx, cy), head_r, color, -1)

    elif cls_id == 3:
        # Vélo : deux cercles + barre
        r = max(3, min(w, h) // 3)
        cy = y2 - r
        cv2.circle(img, (x1 + r, cy), r, color, 2)
        cv2.circle(img, (x2 - r, cy), r, color, 2)
        cv2.line(img, (x1 + r, cy), (x2 - r, cy), color, 2)
        # Cadre
        mid_x = (x1 + x2) // 2
        cv2.line(img, (x1 + r, cy), (mid_x, y1 + r), color, 2)
        cv2.line(img, (x2 - r, cy), (mid_x, y1 + r), color, 2)

    elif cls_id == 4:
        # Voiture : rectangle avec toit arrondi
        roof_h = h // 3
        pts = np.array([
            [x1,          y2],
            [x2,          y2],
            [x2,          y1 + roof_h],
            [x2 - w//5,   y1],
            [x1 + w//5,   y1],
            [x1,          y1 + roof_h],
        ], dtype=np.int32)
        cv2.fillPoly(img, [pts], color)
        # Fenêtres
        win_c = tuple(min(255, c + 60) for c in color)
        cv2.rectangle(img,
                      (x1 + w//5 + 2, y1 + 2),
                      (x2 - w//5 - 2, y1 + roof_h - 2),
                      win_c, -1)

    elif cls_id == 5:
        # Moto : losange + roues
        cx, cy = (x1+x2)//2, (y1+y2)//2
        pts = np.array([[cx, y1], [x2, cy], [cx, y2], [x1, cy]], np.int32)
        cv2.fillPoly(img, [pts], color)

    elif cls_id in (6, 7):
        # Van / camion : rectangle plein avec cabine
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        cab_w = w // 3
        cab_color = tuple(min(255, c + 40) for c in color)
        cv2.rectangle(img, (x1, y1), (x1 + cab_w, y2), cab_color, -1)
        # Fenêtre cabine
        win_c = tuple(min(255, c + 80) for c in color)
        cv2.rectangle(img,
                      (x1 + 2, y1 + 2),
                      (x1 + cab_w - 2, y1 + h//2),
                      win_c, -1)

    elif cls_id == 8:
        # Tricycle : triangle + roue
        pts = np.array([[x1, y2], [x2, y2], [(x1+x2)//2, y1]], np.int32)
        cv2.fillPoly(img, [pts], color)
        cv2.circle(img, ((x1+x2)//2, y2), max(2, h//5), color, -1)

    elif cls_id == 9:
        # Awning-tricycle : toit plat + corps
        roof = h // 3
        cv2.rectangle(img, (x1, y1), (x2, y1 + roof), color, -1)
        body_c = tuple(max(0, c - 40) for c in color)
        cv2.rectangle(img, (x1 + w//6, y1 + roof), (x2 - w//6, y2), body_c, -1)

    else:  # bus (10)
        # Bus : grand rectangle avec rangée de fenêtres
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        win_c = tuple(min(255, c + 80) for c in color)
        n_win = max(2, w // (h // 2 + 1))
        win_w = (w - 4) // max(n_win, 1)
        for i in range(n_win):
            wx1 = x1 + 2 + i * win_w
            wx2 = wx1 + win_w - 2
            cv2.rectangle(img,
                          (wx1, y1 + 2),
                          (wx2, y1 + h // 2 - 1),
                          win_c, -1)

    # Contour léger pour mieux délimiter l'objet
    cv2.rectangle(img, (x1, y1), (x2, y2),
                  tuple(max(0, c - 60) for c in color), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Génération d'une image complète + annotations
# ─────────────────────────────────────────────────────────────────────────────
def generate_image(img_size: int, n_objects_range: tuple,
                   rng: random.Random) -> tuple:
    """
    Retourne (img_array [H,W,3 uint8], annotations [list of dicts]).
    """
    img  = make_background(img_size, rng)
    H = W = img_size

    n_obj = rng.randint(*n_objects_range)
    annotations = []
    placed_boxes = []

    for _ in range(n_obj * 3):   # on essaie 3x pour remplir n_obj
        if len(annotations) >= n_obj:
            break

        cls_id = rng.randint(1, 10)
        size_min, size_max = CLASS_SIZES[cls_id]
        asp_min,  asp_max  = CLASS_ASPECT[cls_id]

        # Taille de l'objet
        obj_h = int(rng.uniform(size_min, size_max) * H)
        obj_h = max(6, obj_h)
        aspect = rng.uniform(asp_min, asp_max)
        obj_w = int(obj_h * aspect)
        obj_w = max(4, min(obj_w, W - 2))

        # Position
        x1 = rng.randint(0, max(0, W - obj_w - 1))
        y1 = rng.randint(0, max(0, H - obj_h - 1))
        x2 = x1 + obj_w
        y2 = y1 + obj_h

        # Vérifie le chevauchement (IoU < 0.4)
        too_much_overlap = False
        for bx1, by1, bx2, by2 in placed_boxes:
            ix1, iy1 = max(x1, bx1), max(y1, by1)
            ix2, iy2 = min(x2, bx2), min(y2, by2)
            inter = max(0, ix2-ix1) * max(0, iy2-iy1)
            area  = obj_w * obj_h
            if area > 0 and inter / area > 0.4:
                too_much_overlap = True
                break
        if too_much_overlap:
            continue

        draw_object(img, cls_id, x1, y1, x2, y2, rng)
        placed_boxes.append((x1, y1, x2, y2))

        annotations.append({
            'x': x1, 'y': y1,
            'w': x2 - x1, 'h': y2 - y1,
            'cls': cls_id,
            'name': CLASSES[cls_id],
        })

    return img, annotations


# ─────────────────────────────────────────────────────────────────────────────
# Sauvegarde au format VisDrone
# ─────────────────────────────────────────────────────────────────────────────
def save_visdrone_annotation(path: Path, annotations: list) -> None:
    """
    Format VisDrone : x,y,w,h,score,cat,trunc,occ  (une ligne par objet)
    score=1 (non ignoré), trunc=0, occ=0
    """
    with open(path, 'w') as f:
        for ann in annotations:
            f.write(f"{ann['x']},{ann['y']},{ann['w']},{ann['h']},"
                    f"1,{ann['cls']},0,0\n")


# ─────────────────────────────────────────────────────────────────────────────
# Prévisualisation : grille d'images
# ─────────────────────────────────────────────────────────────────────────────
def make_preview_grid(img_paths: list, ann_dir: Path,
                      out_path: Path, n_cols: int = 4,
                      thumb_size: int = 200) -> None:
    """
    Crée une grille de n_cols × N images avec les boîtes dessinées.
    Affiche le nom de la classe sur chaque boîte.
    """
    n = min(16, len(img_paths))
    n_rows = math.ceil(n / n_cols)
    grid = np.zeros((n_rows * thumb_size, n_cols * thumb_size, 3), dtype=np.uint8)

    for idx, img_path in enumerate(img_paths[:n]):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Charge les annotations
        ann_path = ann_dir / (Path(img_path).stem + '.txt')
        if ann_path.exists():
            H_img, W_img = img.shape[:2]
            scale_x = thumb_size / W_img
            scale_y = thumb_size / H_img
            img_t = cv2.resize(img, (thumb_size, thumb_size))

            with open(ann_path) as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 6:
                        continue
                    x, y, w, h = map(int, parts[:4])
                    cls_id = int(parts[5])
                    # Redimensionne les coordonnées
                    x1 = int(x * scale_x); y1 = int(y * scale_y)
                    x2 = int((x+w) * scale_x); y2 = int((y+h) * scale_y)
                    color = CLASS_COLORS.get(cls_id, (255, 255, 255))
                    cv2.rectangle(img_t, (x1, y1), (x2, y2), color, 2)
                    # Label
                    label = CLASSES.get(cls_id, str(cls_id))[:4]
                    cv2.putText(img_t, label, (x1, max(0, y1-3)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        else:
            img_t = cv2.resize(img, (thumb_size, thumb_size))

        row = idx // n_cols
        col = idx  % n_cols
        grid[row*thumb_size:(row+1)*thumb_size,
             col*thumb_size:(col+1)*thumb_size] = img_t

    # Sauvegarde
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"  → Preview : {out_path}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description='Génère un dataset synthétique au format VisDrone')
    p.add_argument('--out_dir',    type=str, default='./synthetic_dataset',
                   help='Dossier de sortie')
    p.add_argument('--n_train',    type=int, default=600)
    p.add_argument('--n_val',      type=int, default=150)
    p.add_argument('--img_size',   type=int, default=640)
    p.add_argument('--min_obj',    type=int, default=2,
                   help='Nombre minimum d\'objets par image')
    p.add_argument('--max_obj',    type=int, default=8,
                   help='Nombre maximum d\'objets par image')
    p.add_argument('--seed',       type=int, default=42)
    p.add_argument('--force',      action='store_true',
                   help='Regénère même si le dossier existe déjà')
    return p.parse_args()


def generate_split(split: str, n: int, out_dir: Path,
                   img_size: int, min_obj: int, max_obj: int,
                   seed: int) -> list:
    img_dir = out_dir / 'images'      / split
    ann_dir = out_dir / 'annotations' / split
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    img_paths = []

    print(f"\n  Génération {split} : {n} images ({img_size}×{img_size})",
          flush=True)

    for i in range(n):
        img_array, annotations = generate_image(
            img_size, (min_obj, max_obj), rng
        )
        img_path = img_dir / f'{i:04d}.png'
        ann_path = ann_dir / f'{i:04d}.txt'

        cv2.imwrite(str(img_path),
                    cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        save_visdrone_annotation(ann_path, annotations)
        img_paths.append(img_path)

        if (i + 1) % max(1, n // 10) == 0 or i == n - 1:
            print(f"    {i+1}/{n} images générées", flush=True)

    return img_paths


def main():
    if not CV2_OK:
        print("ERREUR : opencv-python-headless requis.")
        print("  pip install opencv-python-headless")
        return

    args = parse_args()
    out_dir = Path(args.out_dir)

    # Vérifie si déjà généré
    info_path = out_dir / 'dataset_info.json'
    if info_path.exists() and not args.force:
        with open(info_path) as f:
            info = json.load(f)
        print(f"\n✓ Dataset déjà présent dans {out_dir}")
        print(f"  n_train={info['n_train']} | n_val={info['n_val']} | "
              f"img_size={info['img_size']}")
        print(f"  Utilisez --force pour regénérer.")
        return

    print('\n' + '='*60, flush=True)
    print(f'  Génération du dataset synthétique', flush=True)
    print(f'  Dossier  : {out_dir}', flush=True)
    print(f'  Train    : {args.n_train} images', flush=True)
    print(f'  Val      : {args.n_val} images', flush=True)
    print(f'  img_size : {args.img_size}×{args.img_size}', flush=True)
    print(f'  Objets   : {args.min_obj}–{args.max_obj} par image', flush=True)
    print('='*60, flush=True)

    import time
    t0 = time.time()

    # Génère train
    train_paths = generate_split(
        'train', args.n_train, out_dir,
        args.img_size, args.min_obj, args.max_obj,
        seed=args.seed
    )

    # Génère val
    val_paths = generate_split(
        'val', args.n_val, out_dir,
        args.img_size, args.min_obj, args.max_obj,
        seed=args.seed + 1000
    )

    # Grilles de prévisualisation
    print('\n  Création des previews...', flush=True)
    preview_dir = out_dir / 'preview'
    make_preview_grid(
        train_paths,
        out_dir / 'annotations' / 'train',
        preview_dir / 'preview_train.png'
    )
    make_preview_grid(
        val_paths,
        out_dir / 'annotations' / 'val',
        preview_dir / 'preview_val.png'
    )

    # Info JSON
    info = {
        'n_train':   args.n_train,
        'n_val':     args.n_val,
        'img_size':  args.img_size,
        'min_obj':   args.min_obj,
        'max_obj':   args.max_obj,
        'seed':      args.seed,
        'classes':   CLASSES,
        'elapsed_s': round(time.time() - t0, 1),
    }
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)

    elapsed = time.time() - t0
    print(f'\n✓ Dataset généré en {elapsed:.0f}s', flush=True)
    print(f'  Dossier    : {out_dir.resolve()}', flush=True)
    print(f'  Train      : {args.n_train} images', flush=True)
    print(f'  Val        : {args.n_val} images', flush=True)
    print(f'  Previews   : {preview_dir}/', flush=True)
    print(f'\n  Pour lancer l\'entraînement :', flush=True)
    print(f'  python train.py --conv sac --data {out_dir} '
          f'--img_size {args.img_size}', flush=True)
    print(f'  python quick_test.py --data {out_dir}', flush=True)


if __name__ == '__main__':
    main()
