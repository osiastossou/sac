"""
logger.py
=========
Logger complet : console + fichier .log + deux CSV écrits en temps réel :
  - train_batches.csv  : écrit après chaque batch
  - metrics.csv        : résumé par epoch (train + val)
"""

import os
import csv
import logging
from pathlib import Path


def setup_logger(name: str, log_dir: str, conv_name: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(log_dir, f"{conv_name}.log")
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file)
    ch = logging.StreamHandler()
    fmt = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger


class CSVLogger:
    """
    Deux CSV écrits en temps réel avec flush+fsync après chaque ligne.

    train_batches.csv  → epoch, batch, batch_loss, avg_loss, lr, time_s
    metrics.csv        → epoch, train_loss, val_mAP50, val_mAP50_95,
                         lr, train_time_s, val_time_s, total_time_s
    """

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.batch_csv = self.log_dir / 'train_batches.csv'
        self.epoch_csv = self.log_dir / 'metrics.csv'
        self._write(self.batch_csv,
                    [['epoch','batch','batch_loss','avg_loss','lr','time_s']],
                    mode='w')
        self._write(self.epoch_csv,
                    [['epoch','train_loss','val_mAP50','val_mAP50_95',
                      'lr','train_time_s','val_time_s','total_time_s']],
                    mode='w')

    @staticmethod
    def _write(path, rows, mode='a'):
        with open(path, mode, newline='') as f:
            w = csv.writer(f)
            for row in rows:
                w.writerow(row)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass

    # ── Par batch (temps réel) ────────────────────────────────────────────────
    def log_batch(self, epoch, batch, batch_loss, avg_loss, lr, time_s):
        """Appelé après chaque batch — visible immédiatement dans le CSV."""
        self._write(self.batch_csv, [[
            epoch, batch,
            f'{batch_loss:.6f}', f'{avg_loss:.6f}',
            f'{lr:.2e}', f'{time_s:.2f}',
        ]])

    # ── Après train, avant val ────────────────────────────────────────────────
    def log_train_done(self, epoch, train_loss, lr, train_time_s):
        """Écrit la ligne epoch avec PENDING pour les colonnes val."""
        self._write(self.epoch_csv, [[
            epoch, f'{train_loss:.6f}',
            'PENDING', 'PENDING',
            f'{lr:.2e}', f'{train_time_s:.1f}',
            'PENDING', 'PENDING',
        ]])

    # ── Après validation ──────────────────────────────────────────────────────
    def log_epoch(self, epoch, train_loss, val_map50, val_map50_95,
                  lr, train_time_s, val_time_s):
        """Remplace la ligne PENDING par les valeurs complètes."""
        total = train_time_s + val_time_s
        new_row = [
            epoch, f'{train_loss:.6f}',
            f'{val_map50:.4f}', f'{val_map50_95:.4f}',
            f'{lr:.2e}', f'{train_time_s:.1f}',
            f'{val_time_s:.1f}', f'{total:.1f}',
        ]
        with open(self.epoch_csv, 'r', newline='') as f:
            rows = list(csv.reader(f))
        # Remplace la dernière ligne PENDING de cet epoch
        for i in range(len(rows)-1, 0, -1):
            if rows[i] and str(rows[i][0]) == str(epoch) and 'PENDING' in rows[i]:
                rows[i] = new_row
                break
        else:
            rows.append(new_row)
        with open(self.epoch_csv, 'w', newline='') as f:
            csv.writer(f).writerows(rows)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass

    # ── Alias ancien appel ────────────────────────────────────────────────────
    def log(self, epoch, train_loss, val_map50, val_map50_95, lr, elapsed):
        self._write(self.epoch_csv, [[
            epoch, f'{train_loss:.6f}',
            f'{val_map50:.4f}', f'{val_map50_95:.4f}',
            f'{lr:.2e}', f'{elapsed:.1f}', 'N/A', f'{elapsed:.1f}',
        ]])
