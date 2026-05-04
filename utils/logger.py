"""
logger.py
=========
Logger simple : console + fichier CSV + sauvegarde du meilleur modèle.
"""

import os
import csv
import time
import logging
from pathlib import Path


def setup_logger(name: str, log_dir: str, conv_name: str) -> logging.Logger:
    """Configure un logger console + fichier pour un run donné."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(log_dir, f"{conv_name}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Handler fichier
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Handler console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


class CSVLogger:
    """
    Enregistre les métriques epoch par epoch dans un fichier CSV.

    Colonnes : epoch, train_loss, val_mAP50, val_mAP50_95, lr, time_s
    """

    def __init__(self, path: str):
        self.path = path
        self._init()

    def _init(self):
        with open(self.path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_mAP50',
                             'val_mAP50_95', 'lr', 'time_s'])

    def log(self, epoch: int, train_loss: float,
            val_map50: float, val_map50_95: float,
            lr: float, elapsed: float):
        with open(self.path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch,
                             f'{train_loss:.6f}',
                             f'{val_map50:.4f}',
                             f'{val_map50_95:.4f}',
                             f'{lr:.2e}',
                             f'{elapsed:.1f}'])
