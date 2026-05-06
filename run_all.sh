#!/bin/bash
# run_all.sh
# Lance les 11 experiences en sequence puis genere le tableau comparatif.
#
# Usage :
#   chmod +x run_all.sh
#   ./run_all.sh datasets/visdrone.yaml        # avec YAML (telecharge auto)
#   ./run_all.sh /path/to/visdrone             # avec dossier direct
#   ./run_all.sh datasets/visdrone.yaml 50 8 640
#
# Sur Google Colab :
#   import subprocess
#   subprocess.run(['bash', 'run_all.sh', 'datasets/visdrone.yaml'])

DATA=${1:-"datasets/visdrone.yaml"}
EPOCHS=${2:-50}
BATCH=${3:-8}
IMG_SIZE=${4:-640}
OUT_DIR="./runs"

CONVS=(
  "standard"
  "deformable"
  "dynamic_filter"
  "dynamic_conv"
  "condconv"
  "pac"
  "knconv"
  "hyperconv"
  "sac"
  "sac_fast"
  "pwc"
  "dpwc"
)

echo "======================================================"
echo "  PWC Benchmark - Object Detection"
echo "  Data     : $DATA"
echo "  Epochs   : $EPOCHS"
echo "  Batch    : $BATCH"
echo "  img_size : $IMG_SIZE"
echo "  Out      : $OUT_DIR"
echo "  Convs    : ${#CONVS[@]} operators"
echo "======================================================"

TOTAL=${#CONVS[@]}
IDX=0

for CONV in "${CONVS[@]}"; do
  IDX=$((IDX + 1))
  echo ""
  echo "------------------------------------------------------"
  echo "  [$IDX/$TOTAL] Training : $CONV"
  echo "------------------------------------------------------"
  python train.py \
    --conv     "$CONV" \
    --data     "$DATA" \
    --epochs   "$EPOCHS" \
    --batch    "$BATCH" \
    --img_size "$IMG_SIZE" \
    --out_dir  "$OUT_DIR" \
    --workers  2

  if [ $? -ne 0 ]; then
    echo "  WARNING: $CONV failed, continuing..."
  fi
done

echo ""
echo "======================================================"
echo "  Final evaluation - comparison table"
echo "======================================================"
python evaluate.py \
  --data     "$DATA" \
  --out_dir  "$OUT_DIR" \
  --batch    "$BATCH" \
  --img_size "$IMG_SIZE" \
  --workers  2
