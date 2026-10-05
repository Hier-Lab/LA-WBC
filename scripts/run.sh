#!/usr/bin/env bash
#
# End-to-end pipeline for each dataset subset:
#   (1) preprocess raw episodes -> processed_dataset.h5
#   (2) compute the locomotion cost map
#   (3) train all models (baselines + alpha-scaled RecGNN)
#
# Expects the raw data under <repo>/dataset/<subset>/<run>/ (see scripts/download_dataset.py).
# Optional environment variables:
#   SUBSETS="w_cont_id" GPU=1 scripts/run.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR="${DATASET_DIR:-$REPO_ROOT/dataset}"
PYTHON="${PYTHON:-python}"
SUBSETS="${SUBSETS:-w_cont_id wo_cont_id}"
GPU="${GPU:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"

cd "$REPO_ROOT"

for sub in $SUBSETS; do
  sub_dir="$DATASET_DIR/$sub"
  if [ ! -d "$sub_dir" ]; then
    echo "[skip] subset not found: $sub_dir"
    continue
  fi

  echo "=================================================="
  echo " SUBSET: $sub  ->  results/$sub"
  echo "=================================================="

  # A "run folder" is any directory that directly contains an "episodes/" dir.
  while IFS= read -r ep_dir; do
    run_dir="$(dirname "$ep_dir")"

    # ---- 1. Preprocess (skipped if already done) ---------------------------
    if [ -f "$run_dir/processed_dataset.h5" ]; then
      echo "[preprocess] already done: $run_dir/processed_dataset.h5"
    else
      "$PYTHON" scripts/process_dataset.py --src "$run_dir"
    fi

    # ---- 2. Cost map -------------------------------------------------------
    npz="results/cost_maps/cost_map_${sub}_$(basename "$run_dir").npz"
    "$PYTHON" scripts/compute_cost_map.py --data "$run_dir" --output "$npz"
  done < <(find "$sub_dir" -type d -name episodes)

  # ---- 3. Train ------------------------------------------------------------
  "$PYTHON" scripts/train.py --data_dir "$sub_dir" --out_dir "$sub" \
    --gpu_id "$GPU" --num_workers "$NUM_WORKERS"
done

echo "All done"
