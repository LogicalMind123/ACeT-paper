#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run.sh demo|viscosity|clearance|clinical|all

TASK="${1:-demo}"
echo "[run] Task: $TASK"

# Ensure a non-GUI backend for matplotlib (stable on Windows/macOS/Linux)
export MPLBACKEND=Agg

mkdir -p results

case "$TASK" in
  # ------------------------------------------------------------
  # Demo = clearance parity (small, fast)
  # ------------------------------------------------------------
  demo)
    mkdir -p results/demo

    python src/shared/clearance/panel_A_B_parity.py \
      --train_file src/shared/clearance/clearance_train.csv \
      --test_file  src/shared/clearance/clearance_test.csv \
      --task regression \
      --head_type rbf

    # Safely collect demo artifacts into results/demo/
    shopt -s nullglob
    files=(clearance_*)
    if ((${#files[@]})); then
      mv -f "${files[@]}" results/demo/
      echo "[run] Moved ${#files[@]} files to results/demo/"
    else
      echo "[warn] No files found matching clearance_*"
      echo "[hint] If your outputs don't start with 'clearance_', adjust the pattern above."
    fi

    # Keep a reviewer-friendly snapshot
    mkdir -p results/expected_demo
    cp -f results/demo/* results/expected_demo/ 2>/dev/null || true
    echo "[run] Copied demo outputs to results/expected_demo/"
    ;;

  # ------------------------------------------------------------
  # Full viscosity pipeline
  # ------------------------------------------------------------
  viscosity)
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

    shopt -s nullglob
    files=(viscosity_*)
    ((${#files[@]})) && mv -f "${files[@]}" results/viscosity/ || echo "[warn] No viscosity_* files to move"
    ;;

  # ------------------------------------------------------------
  # Full clearance pipeline
  # ------------------------------------------------------------
  clearance)
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

    shopt -s nullglob
    files=(clearance_*)
    ((${#files[@]})) && mv -f "${files[@]}" results/clearance/ || echo "[warn] No clearance_* files to move"
    ;;

  # ------------------------------------------------------------
  # Full clinical (regulatory success) pipeline
  # ------------------------------------------------------------
  clinical)
    mkdir -p results/clinical

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

echo ""
read -p "Press Enter to close..."

  # ------------------------------------------------------------
  # Run everything
  # ------------------------------------------------------------
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

echo ""
read -p "Press Enter to close..."
