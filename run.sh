#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run.sh demo
#   bash run.sh viscosity
#   bash run.sh clearance
#   bash run.sh clinical
#   bash run.sh all

TASK="${1:-demo}"
echo "[run] Task: $TASK"

mkdir -p results

case "$TASK" in
  demo)
    mkdir -p results/demo
    export MPLBACKEND=Agg
    python src/shared/clearance/panel_A_B_parity.py \
      --train_file src/shared/clearance/clearance_train.csv \
      --test_file  src/shared/clearance/clearance_test.csv \
      --task regression \
      --head_type rbf
    shopt -s nullglob
    files=(viscosity_*)
    if ((${#files[@]})); then
      mv -f "${files[@]}" results/demo/
      echo "[run] Moved ${#files[@]} files to results/demo/"
    else
      echo "[warn] No files found matching viscosity_*"
    fi
    ;;

  viscosity)
    mkdir -p results/viscosity
    export MPLBACKEND=Agg
    python src/shared/viscosity/panel_A_D_parity_featureimp.py \
      --train_file src/shared/viscosity/antibodies_train.csv \
      --test_file  src/shared/viscosity/antibodies_test.csv \
      --task regression \
      --head_type kan
    python src/shared/viscosity/panel_B_bootstrap.py \
      --train_file src/shared/viscosity/antibodies_train.csv \
      --test_file  src/shared/viscosity/antibodies_test.csv \
      --task regression \
      --head_type kan
    python src/shared/viscosity/panel_C_baselines.py \
      --train_file src/shared/viscosity/antibodies_train.csv \
      --test_file  src/shared/viscosity/antibodies_test.csv \
      --task regression \
      --head_type kan
    shopt -s nullglob
    files=(viscosity_*)
    ((${#files[@]})) && mv -f "${files[@]}" results/viscosity/ || echo "[warn] No viscosity_* files to move"
    ;;

  clearance)
    mkdir -p results/clearance
    export MPLBACKEND=Agg
    python src/shared/clearance/panel_A_B_parity.py \
      --train_file src/shared/clearance/clearance_train.csv \
      --test_file  src/shared/clearance/clearance_test.csv \
      --task regression \
      --head_type rbf
    python src/shared/clearance/panel_C_baselines.py \
      --train_file src/shared/clearance/clearance_train.csv \
      --test_file  src/shared/clearance/clearance_test.csv \
      --task regression \
      --head_type rbf
    python src/shared/clearance/panel_D_shap.py \
      --train_file src/shared/clearance/clearance_train.csv \
      --test_file  src/shared/clearance/clearance_test.csv \
      --task regression \
      --head_type rbf
    shopt -s nullglob
    files=(clearance_*)
    ((${#files[@]})) && mv -f "${files[@]}" results/clearance/ || echo "[warn] No clearance_* files to move"
    ;;

  clinical)
    mkdir -p results/clinical
    export MPLBACKEND=Agg
    python src/shared/clinical/panels.py \
      --train_file src/shared/clinical/InternalCohort_112mAbs_train.csv \
      --test_file  src/shared/clinical/InternalCohort_112mAbs_test.csv \
      --external_file ExternalCohort_14mAbs.csv \
      --status_col Updated.Status \
      --task classification \
      --head_type mlp
    shopt -s nullglob
    files=(cm_*.png Table_S1_thresholds.csv)
    ((${#files[@]})) && mv -f "${files[@]}" results/clinical/ || echo "[warn] No clinical outputs to move"
    ;;

  all)
    bash "$0" viscosity
    bash "$0" clearance
    bash "$0" clinical
    ;;

  *)
    echo "Usage: bash run.sh [demo|viscosity|clearance|clinical|all]"
    exit 1
    ;;
esac

echo "[run] Done. See results/"

