"""
colab_launcher.py
=================
Script de lancement optimisé pour Google Colab / Kaggle.
Copiez et exécutez ce fichier dans une cellule Colab.

Étapes :
  1. Monte Google Drive (ou Kaggle dataset)
  2. Installe les dépendances
  3. Lance les 9 expériences
  4. Affiche le tableau comparatif final
"""

# ── Cellule 1 : Installation ─────────────────────────────────────────────────
INSTALL = """
pip install torch torchvision --quiet
pip install opencv-python-headless --quiet
# torchvision est nécessaire pour deform_conv2d (DeformableConv)
"""

# ── Cellule 2 : Téléchargement VisDrone ──────────────────────────────────────
DOWNLOAD = """
# Option A — depuis Kaggle (recommandé sur Kaggle)
# !kaggle datasets download -d bulentsiyah/visdrone-dataset
# !unzip visdrone-dataset.zip -d /content/visdrone

# Option B — depuis Google Drive (si vous avez déjà le dataset)
# from google.colab import drive
# drive.mount('/content/drive')
# VISDRONE_ROOT = '/content/drive/MyDrive/visdrone'

# Structure attendue :
# /content/visdrone/
#     images/train/*.jpg
#     images/val/*.jpg
#     annotations/train/*.txt
#     annotations/val/*.txt

VISDRONE_ROOT = '/content/visdrone'  # ← adaptez ce chemin
"""

# ── Cellule 3 : Cloner / uploader le code ────────────────────────────────────
UPLOAD = """
# Si vous avez uploadé sac_experiment.zip dans Colab :
# !unzip sac_experiment.zip -d /content/
# %cd /content/sac_experiment

# Sinon, si le code est sur GitHub :
# !git clone https://github.com/votre_user/sac_experiment /content/sac_experiment
# %cd /content/sac_experiment
"""

# ── Cellule 4 : Lancement d'une seule convolution (test rapide) ───────────────
SINGLE_RUN = """
import subprocess, sys
sys.path.insert(0, '/content/sac_experiment')

result = subprocess.run([
    'python', 'train.py',
    '--conv',     'sac',
    '--data',     VISDRONE_ROOT,
    '--epochs',   '5',           # ← augmentez à 50 pour les vrais résultats
    '--batch',    '8',
    '--img_size', '640',
    '--out_dir',  './runs',
    '--workers',  '2',
], capture_output=False)
"""

# ── Cellule 5 : Lancement de toutes les 9 convolutions ───────────────────────
ALL_RUNS = """
from models import CONV_REGISTRY
import subprocess, sys
sys.path.insert(0, '/content/sac_experiment')

VISDRONE_ROOT = '/content/visdrone'   # ← adaptez
EPOCHS        = 50
BATCH         = 8
IMG_SIZE      = 640

for conv_name in CONV_REGISTRY:
    print(f'\\n{"="*60}')
    print(f'  Entraînement : {conv_name.upper()}')
    print(f'{"="*60}')
    subprocess.run([
        'python', 'train.py',
        '--conv',     conv_name,
        '--data',     VISDRONE_ROOT,
        '--epochs',   str(EPOCHS),
        '--batch',    str(BATCH),
        '--img_size', str(IMG_SIZE),
        '--out_dir',  './runs',
        '--workers',  '2',
    ])
"""

# ── Cellule 6 : Tableau comparatif ───────────────────────────────────────────
EVALUATE = """
import subprocess
subprocess.run([
    'python', 'evaluate.py',
    '--data',     VISDRONE_ROOT,
    '--out_dir',  './runs',
    '--batch',    '8',
    '--img_size', '640',
])
"""

# ── Cellule 7 : Visualisation des courbes ────────────────────────────────────
PLOT = """
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

runs_dir = Path('./runs')
conv_names = list(CONV_REGISTRY.keys())
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for conv in conv_names:
    csv_path = runs_dir / conv / 'metrics.csv'
    if not csv_path.exists():
        continue
    df = pd.read_csv(csv_path)
    label = conv.upper()
    style = dict(linewidth=2.5, linestyle='-') if conv == 'sac' \
            else dict(linewidth=1.2, linestyle='--')
    axes[0].plot(df['epoch'], df['val_mAP50'],    label=label, **style)
    axes[1].plot(df['epoch'], df['val_mAP50_95'], label=label, **style)

for ax, title in zip(axes, ['mAP@50', 'mAP@50:95']):
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Epoch')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

plt.suptitle('Comparaison des 9 opérateurs de convolution — VisDrone', fontsize=14)
plt.tight_layout()
plt.savefig('./runs/comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print('Figure sauvegardée : ./runs/comparison.png')
"""

if __name__ == '__main__':
    print("Copiez les blocs ci-dessus dans des cellules Colab séparées.")
    print("Ordre d'exécution :")
    print("  1. INSTALL    — pip install")
    print("  2. DOWNLOAD   — préparer VisDrone")
    print("  3. UPLOAD     — mettre en place le code")
    print("  4. SINGLE_RUN — test rapide (5 epochs) avec SAC")
    print("  5. ALL_RUNS   — lancer les 9 expériences complètes")
    print("  6. EVALUATE   — tableau comparatif")
    print("  7. PLOT       — courbes de convergence")
