# ACeT: Transformer model for mAb viscosity, PK, and regulatory success

This repository contains the code and small example data to reproduce the analyses/figures in our NBME submission.

---

# 1) System requirements
- OS tested: Windows 11 Pro (23H2)
- Python: 3.11.7
- Dependencies (and versions): see [requirements.txt](./requirements.txt)
- Hardware: Runs on a normal desktop/laptop (CPU). GPU optional for speed; no special hardware required.

# 2) Installation guide
# Create & activate an isolated environment (recommended):
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

Typical install time on a "normal" desktop: ~ 15 minutes on common broadband.

# 3) Demo (small data)
This demo runs the clearance parity analysis on the small CSVs included in the repository.
Run: bash run.sh demo

Expected output: figures and CSVs under results/demo/ (e.g., clearance_parity_loglog.png, clearance_error_cdf.png, (any other clearance_* files produced by the script))
Expected runtime on a normal desktop (CPU): ~2-5 minutes.
Expected outputs folder: We also include pre-generated demo outputs in results/expected_demo/ so reviewers can preview example figures and tables without running the code.
Note on repository size: Only results/expected_demo/ is tracked in Git. All other results/ outputs are ignored to keep the repository light.

# 4) Instructions for use (full reproduction)
Reproduce each analysis:

# Viscosity
bash run.sh viscosity
# Clearance (PK proxy)
bash run.sh clearance
# Regulatory success (clinical classification)
bash run.sh clinical

Reproduction instructions: Seeds are set inside each script; configs/flags are documented at the top of the scripts. Task-specific datasets are included in the repository so the demo runs without restrictions.
Typical runtimes (CPU):
- viscosity: ~5-10 minutes
- clearance: ~5-10 minutes
- clinical: ~2-5 minutes

Code overview / pseudocode
Overall pipeline and figure-generation logic lives in:

- Viscosity: src/shared/viscosity/*.py
- Clearance: src/shared/clearance/*.py
- Clinical: src/shared/clinical/*.py (entry: panels.py)

Each panel script trains/evaluates and writes plots/tables to results/<task>/.


# Quick start (full setup & run)
git clone https://github.com/LogicalMind123/AceT-paper.git
cd AceT-paper

# Create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run the demo (clearance parity)
bash run.sh demo
ls -la results/demo

# Run all analyses (full reproduction)
bash run.sh all
# Check results in: results/viscosity/, results/clearance/, results/clinical/
