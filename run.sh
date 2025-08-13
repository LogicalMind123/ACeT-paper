#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   bash run.sh demo
#   bash run.sh viscosity
#   bash run.sh clearance
#   bash run.sh clinical
#   bash run.sh all

TASK="${1:-demo}"

echo "[run] Task: $TASK"

case "$TASK" in
  demo)
    # small demo on included sample data; writes to results/demo/
    python src/shared/viscosity/panel_A_D_parity_featureimp.py \
      --train_file src/shared/viscosity/antibodies_train.csv \
      --test_file  src/shared/viscosity/antibodies_test.csv \
      --task regression \
      --head_type kan
    ;;

  viscosity)
    # run all viscosity panels; ensure each script writes to /results/viscosity/
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
    ;;

  clearance)
    # ensure these scripts save outputs under results/clearance/
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
    ;;

  clinical)
    # runs classification panels
    python src/shared/clinical/panels.py \
      --train_file src/shared/clinical/InternalCohort_112mAbs_train.csv \
      --test_file  src/shared/clinical/InternalCohort_112mAbs_test.csv \
      --external_file ExternalCohort_14mAbs.csv
      --status_col Updated.Status \
      --task classification \
      --head_type mlp
    ;;

  all)
    bash "$0" viscosity
    bash "$0" clearance
    bash "$0" clinical
    ;;

  *)
    echo "Unknown task: $TASK"
    exit 1
    ;;
esac

echo "[run] Done. See results/"
