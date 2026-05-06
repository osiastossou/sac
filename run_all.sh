#!/bin/bash
# run_all.sh
# Lance les 9 expériences en séquence puis génère le tableau comparatif.
#
# Usage :
#   chmod +x run_all.sh
#   ./run_all.sh /path/to/visdrone
#
# Sur Google Colab, appelez directement depuis Python :
#   import subprocess
#   subprocess.run(['bash', 'run_all.sh', '/path/to/visdrone'])

DATA=${1:-"/content/visdrone"}
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
  "pwc"
)

echo "======================================================"
echo "  SAC Benchmark — VisDrone"
echo "  Data     : $DATA"
echo "  Epochs   : $EPOCHS"
echo "  Batch    : $BATCH"
echo "  img_size : $IMG_SIZE"
echo "  Out      : $OUT_DIR"
echo "======================================================"

for CONV in "${CONVS[@]}"; do
  echo ""
  echo "------------------------------------------------------"
  echo "  Training : $CONV"
  echo "------------------------------------------------------"
  python train.py \
    --conv     "$CONV" \
    --data     "$DATA" \
    --epochs   "$EPOCHS" \
    --batch    "$BATCH" \
    --img_size "$IMG_SIZE" \
    --out_dir  "$OUT_DIR" \
    --workers  2
done

echo ""
echo "======================================================"
echo "  Évaluation finale — tableau comparatif"
echo "======================================================"
python evaluate.py \
  --data    "$DATA" \
  --out_dir "$OUT_DIR" \
  --batch   "$BATCH" \
  --img_size "$IMG_SIZE" \
  --workers 2
