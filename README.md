# SAC Experiment — VisDrone Benchmark

Comparaison des 9 opérateurs de convolution sur VisDrone-DET.

## Commandes recommandées (Google Colab Pro)

### Test rapide (1 conv, 3 epochs)
```bash
python train.py --conv sac \
  --data /content/drive/MyDrive/.../VisDrone \
  --epochs 3 --batch 4 --img_size 320
```

### Benchmark complet (toutes les convolutions)
```bash
bash run_all.sh /content/drive/MyDrive/.../VisDrone 50 4 320
```

### Paramètres recommandés par config GPU
| GPU Colab      | img_size | batch | SAC ~ms/batch |
|----------------|----------|-------|---------------|
| T4 (gratuit)   | 320      | 2     | ~8ms          |
| T4 (gratuit)   | 416      | 2     | ~15ms         |
| A100 (Pro+)    | 640      | 4     | ~12ms         |

## Pourquoi SAC est plus lent que StandardConv ?
SAC génère un noyau différent pour chaque position spatiale (H'×W' noyaux par image).
Sur GPU, cela reste raisonnable car toutes les opérations sont vectorisées.
Sur CPU, évitez img_size > 320.

## Structure
```
sac_experiment/
├── models/convolutions.py  # 9 opérateurs
├── models/detector.py      # TinyDetector (FPN léger)
├── data/visdrone.py        # Dataloader VisDrone
├── utils/metrics.py        # mAP@50 et mAP@50:95
├── utils/logger.py         # Logs CSV
├── train.py                # Entraînement
├── evaluate.py             # Tableau comparatif
├── run_all.sh              # Lance les 9 expériences
└── colab_launcher.py       # Guide cellule par cellule
```

## Format dataset VisDrone attendu
```
VisDrone/
├── images/
│   ├── train/  *.jpg
│   └── val/    *.jpg
└── annotations/
    ├── train/  *.txt
    └── val/    *.txt
```
