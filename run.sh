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
    python src/shared/viscosity/panel_A_D_parity_featureimp.py \
      --train_file src/shared/viscosity/antibodies_train.csv \
      --test_file  src/shared/viscosity/antibodies_test.csv \
      --task regression \
      --head_type kan
    # collect likely outputs from viscosity demo
    mv -f viscosity_* results/demo/ 2>/dev/null || true
    ;;

  viscosity)     # run all viscosity panels; ensure each script writes to /results/viscosity/
    mkdir -p results/viscosity
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
    mv -f viscosity_* results/viscosity/ 2>/dev/null || true
    ;;

  clearance)  # ensure these scripts save outputs under results/clearance/
    mkdir -p results/clearance
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
    mv -f clearance_* results/clearance/ 2>/dev/null || true
    ;;

  clinical)
    mkdir -p results/clinical
    python src/shared/clinical/panels.py \
      --train_file src/shared/clinical/InternalCohort_112mAbs_train.csv \
      --test_file  src/shared/clinical/InternalCohort_112mAbs_test.csv \
      --external_file ExternalCohort_14mAbs.csv
      --status_col Updated.Status \
      --task classification \
      --head_type mlp
    mv -f cm_*.png Table_S1_thresholds.csv results/clinical/ 2>/dev/null || true
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

