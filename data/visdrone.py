# visdrone.py - Universal dataset loader for object detection
# Supports YAML config files (YOLO-style) and direct folder paths
# Auto-detects folder structure and annotation format

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VISDRONE_VALID_CLASSES = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
CLASS_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9}
NUM_CLASSES = 10
VISDRONE_CLASS_NAMES = [
    'pedestrian', 'people', 'bicycle', 'car', 'van',
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
]

_SPLIT_ALIASES = {
    'train': ['train'],
    'val':   ['val', 'valid', 'validation'],
}
_IMG_DIRS = ['images', 'imgs', 'image']
_ANN_DIRS = ['annotations', 'labels', 'label', 'annotation']
_IMG_EXTS = ['*.jpg', '*.jpeg', '*.png', '*.bmp']


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------
def load_yaml(yaml_path: str) -> dict:
    """Load a YAML dataset config file (YOLO/Ultralytics-style).
    
    If the dataset root does not exist and the YAML contains a 'download'
    field, executes the download script automatically.
    """
    import yaml
    yaml_path = Path(yaml_path)
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    # Resolve root path - relative to YAML file location if not absolute
    root = Path(cfg.get('path', '.'))
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    cfg['_root'] = root

    # Number of classes from 'nc' or length of 'names'
    if 'nc' not in cfg and 'names' in cfg:
        cfg['nc'] = len(cfg['names'])

    # Auto-download if dataset missing and 'download' key present
    if not root.exists() and 'download' in cfg:
        _run_download(cfg, root, yaml_path)

    return cfg


def _run_download(cfg: dict, root: Path, yaml_path: Path):
    """Execute the download script embedded in the YAML config."""
    download_script = cfg['download']

    print(f"\n[Dataset] Root '{root}' not found.", flush=True)
    print(f"[Dataset] Running download script from {yaml_path.name}...",
          flush=True)

    root.mkdir(parents=True, exist_ok=True)

    # Expose 'yaml' variable as Ultralytics does
    yaml_ctx = dict(cfg)
    yaml_ctx['path'] = str(root)

    exec_globals = {
        '__builtins__': __builtins__,
        'yaml': yaml_ctx,
    }

    try:
        exec(download_script, exec_globals)
        print(f"[Dataset] Download complete -> {root}", flush=True)
    except Exception as e:
        print(f"[Dataset] Download failed: {e}", flush=True)
        print(f"[Dataset] Please download manually and place in: {root}",
              flush=True)
        raise


def parse_data_source(source: str) -> dict:
    """
    Accept either a .yaml file or a folder path.
    Returns a normalized config dict with keys: _root, train, val, nc, names
    """
    p = Path(source)
    if p.suffix in ('.yaml', '.yml'):
        return load_yaml(source)

    # Direct folder mode - build minimal config
    return {
        '_root': p.resolve(),
        'train': None,
        'val':   None,
        'nc':    NUM_CLASSES,
        'names': {i: n for i, n in enumerate(VISDRONE_CLASS_NAMES)},
    }


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
def _find_dir(parent: Path, candidates: List[str]) -> Optional[Path]:
    """Return first existing subdirectory among candidates."""
    for name in candidates:
        p = parent / name
        if p.is_dir():
            return p
    return None


def _resolve_paths(root: Path, split: str,
                   yaml_hint: Optional[str] = None) -> Tuple[Path, Path]:
    """
    Resolve (img_dir, ann_dir) for a given split.

    Supported structures:
      A) root/images/train/  + root/annotations/train/
         root/images/val/    + root/labels/val/
      B) root/train/images/  + root/train/labels/
         root/val/images/    + root/val/annotations/

    If yaml_hint is provided (e.g. 'images/train'), uses it to find img_dir
    then searches for ann_dir automatically.
    """
    split_aliases = _SPLIT_ALIASES.get(split, [split])

    # --- YAML hint: use explicit path from config ----------------------------
    if yaml_hint:
        img_dir = root / yaml_hint
        if img_dir.is_dir():
            hint_parts = list(Path(yaml_hint).parts)
            # Try replacing image-dir component with annotation-dir name
            for ann_name in _ANN_DIRS:
                new_parts = []
                replaced = False
                for part in hint_parts:
                    if not replaced and any(part.startswith(d) for d in _IMG_DIRS):
                        new_parts.append(ann_name)
                        replaced = True
                    else:
                        new_parts.append(part)
                if new_parts:
                    ann_dir = root / Path(*new_parts)
                    if ann_dir.is_dir():
                        return img_dir, ann_dir
            # Fallback: look for labels/ sibling of the split folder
            for ann_name in _ANN_DIRS:
                ann_dir = root / ann_name / hint_parts[-1]
                if ann_dir.is_dir():
                    return img_dir, ann_dir

    # --- Structure A: root/images/train + root/annotations/train ------------
    img_base = _find_dir(root, _IMG_DIRS)
    if img_base:
        img_dir = _find_dir(img_base, split_aliases)
        if img_dir:
            ann_base = _find_dir(root, _ANN_DIRS)
            if ann_base:
                ann_dir = _find_dir(ann_base, split_aliases)
                if ann_dir:
                    return img_dir, ann_dir

    # --- Structure B: root/train/images + root/train/labels -----------------
    split_dir = _find_dir(root, split_aliases)
    if split_dir:
        img_dir = _find_dir(split_dir, _IMG_DIRS)
        ann_dir = _find_dir(split_dir, _ANN_DIRS)
        if img_dir and ann_dir:
            return img_dir, ann_dir

    # --- Nothing found -------------------------------------------------------
    raise FileNotFoundError(
        f"\nNo dataset structure found in '{root}' for split='{split}'.\n"
        f"Supported structures:\n"
        f"  A)  root/images/train/  + root/annotations/train/\n"
        f"      root/images/val/    + root/labels/val/\n"
        f"  B)  root/train/images/  + root/train/labels/\n"
        f"      root/val/images/    + root/val/annotations/\n"
        f"  YAML: specify train/val paths in your .yaml file\n"
        f"  Accepted aliases: val/valid/validation, labels/annotations"
    )


# ---------------------------------------------------------------------------
# Annotation format detection
# ---------------------------------------------------------------------------
def _detect_format(ann_path: Path) -> str:
    """
    Detect annotation format:
      - 'visdrone' : x,y,w,h,score,cat,trunc,occ  (>=6 comma-sep cols, pixel coords)
      - 'yolo'     : cls xc yc w h                 (5 space-sep cols, normalized [0,1])
    """
    with open(ann_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            # YOLO: 5 columns, all values in [0,1] except first
            if len(parts) == 5:
                try:
                    vals = [float(p) for p in parts[1:]]
                    if all(0.0 <= v <= 1.0 for v in vals):
                        return 'yolo'
                except ValueError:
                    pass
            # VisDrone: >=6 comma-separated columns
            parts_c = line.strip().split(',')
            if len(parts_c) >= 6:
                return 'visdrone'
    return 'visdrone'


# ---------------------------------------------------------------------------
# Universal Dataset
# ---------------------------------------------------------------------------
class VisDroneDataset(Dataset):
    """
    Universal object detection dataset.

    Accepts either a YAML config file or a root folder path.
    Auto-detects folder structure (A or B) and annotation format
    (VisDrone or YOLO).

    YAML format (recommended):
        path: /path/to/dataset
        train: images/train
        val:   images/val
        nc:    10
        names:
          0: pedestrian
          ...

    Args:
        source   : path to .yaml config OR root dataset folder
        split    : 'train' or 'val'
        img_size : resize target (square)
        max_dets : max detections per image (for padding)
    """

    def __init__(self, source: str, split: str = 'train',
                 img_size: int = 640, max_dets: int = 500):
        self.img_size   = img_size
        self.max_dets   = max_dets
        self.ann_format = None

        # Load config
        cfg         = parse_data_source(source)
        self.cfg    = cfg
        self.root   = cfg['_root']
        self.nc     = cfg.get('nc', NUM_CLASSES)
        self.names  = cfg.get('names',
                              {i: n for i, n in enumerate(VISDRONE_CLASS_NAMES)})

        # Resolve paths
        yaml_hint = cfg.get(split)
        self.img_dir, self.ann_dir = _resolve_paths(self.root, split, yaml_hint)

        # Collect (image, annotation) pairs
        self.samples = []
        for ext in _IMG_EXTS:
            for img_path in sorted(self.img_dir.glob(ext)):
                ann_path = self.ann_dir / (img_path.stem + '.txt')
                if ann_path.exists():
                    self.samples.append((img_path, ann_path))

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No annotated images found.\n"
                f"  Images : {self.img_dir}\n"
                f"  Labels : {self.ann_dir}"
            )

        # Detect annotation format
        for _, ann_path in self.samples[:5]:
            if ann_path.stat().st_size > 0:
                self.ann_format = _detect_format(ann_path)
                break
        if not self.ann_format:
            self.ann_format = 'visdrone'

        print(f"  [{split}] {len(self.samples)} images | "
              f"format={self.ann_format} | nc={self.nc} | "
              f"img={self.img_dir.relative_to(self.root)}",
              flush=True)

    def __len__(self):
        return len(self.samples)

    # --- Annotation parsers -------------------------------------------------
    def _parse_visdrone(self, ann_path: Path,
                        orig_w: int, orig_h: int) -> np.ndarray:
        boxes = []
        with open(ann_path) as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                try:
                    x, y, w, h = map(float, parts[:4])
                    cat = int(parts[5])
                except (ValueError, IndexError):
                    continue
                if cat not in VISDRONE_VALID_CLASSES or w <= 0 or h <= 0:
                    continue
                boxes.append([
                    (x + w/2) / orig_w,
                    (y + h/2) / orig_h,
                    w / orig_w,
                    h / orig_h,
                    CLASS_MAP[cat]
                ])
        return (np.array(boxes, np.float32) if boxes
                else np.zeros((0, 5), np.float32))

    def _parse_yolo(self, ann_path: Path) -> np.ndarray:
        boxes = []
        with open(ann_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])
                except (ValueError, IndexError):
                    continue
                if w <= 0 or h <= 0:
                    continue
                boxes.append([xc, yc, w, h, cls_id % self.nc])
        return (np.array(boxes, np.float32) if boxes
                else np.zeros((0, 5), np.float32))

    # --- __getitem__ --------------------------------------------------------
    def __getitem__(self, idx):
        img_path, ann_path = self.samples[idx]

        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        boxes = (self._parse_yolo(ann_path) if self.ann_format == 'yolo'
                 else self._parse_visdrone(ann_path, orig_w, orig_h))

        img = cv2.resize(img, (self.img_size, self.img_size))
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img  = (img - mean) / std

        n = min(len(boxes), self.max_dets)
        target = np.zeros((self.max_dets, 5), np.float32)
        if n > 0:
            target[:n] = boxes[:n]
        mask = torch.zeros(self.max_dets, dtype=torch.bool)
        mask[:n] = True

        return {
            'image':     img,
            'boxes':     torch.from_numpy(target),
            'mask':      mask,
            'img_path':  str(img_path),
            'orig_size': (orig_h, orig_w),
        }


# ---------------------------------------------------------------------------
# Collate and builder
# ---------------------------------------------------------------------------
def collate_fn(batch):
    return {
        'image':     torch.stack([b['image']  for b in batch]),
        'boxes':     torch.stack([b['boxes']  for b in batch]),
        'mask':      torch.stack([b['mask']   for b in batch]),
        'img_path':  [b['img_path']  for b in batch],
        'orig_size': [b['orig_size'] for b in batch],
    }


def build_dataloaders(source: str,
                      img_size: int = 640,
                      batch_size: int = 8,
                      num_workers: int = 4,
                      max_dets: int = 500):
    """
    Build train and val dataloaders.

    Args:
        source : path to a .yaml config file  OR  a dataset root folder

    Returns:
        (train_loader, val_loader)
    """
    print(f"\n[Dataset] {source}", flush=True)
    train_ds = VisDroneDataset(source, 'train', img_size, max_dets)
    val_ds   = VisDroneDataset(source, 'val',   img_size, max_dets)

    train_dl = DataLoader(train_ds, batch_size, shuffle=True,
                          num_workers=num_workers, collate_fn=collate_fn,
                          pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size, shuffle=False,
                          num_workers=num_workers, collate_fn=collate_fn,
                          pin_memory=True)
    return train_dl, val_dl
