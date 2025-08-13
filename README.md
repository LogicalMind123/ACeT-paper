# ACeT: Transformer model for mAb viscosity, PK, and regulatory success

This repository contains the code and small example data to reproduce the analyses/figures in our NBME submission.

---

1) System requirements
- OS tested: Windows 11 Pro (23H2)
- Python: 3.11.7
- Dependencies (and versions): listed in (./requirements.txt)
- Hardware: Runs on a normal desktop/laptop (CPU). GPU optional for speed; no special hardware required.

2) Installation guide
# Create & activate an isolated environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

Typical install time on a "normal" desktop: ~ 15 minutes on common broadband.

3) Demo (small data)
This demo uses the small sample CSVs included in the repo. It generates example figures/metrics.
Run: bash run.sh demo
Expected output: figures and metrics.json written to results/ (subfolders per task).
Expected run time on a normal desktop: ~ 2 minutes CPU.

4) Instructions for use (full reproduction)
Reproduce each analysis:

# Viscosity
bash run.sh viscosity
# Clearance (PK proxy)
bash run.sh clearance
# Regulatory success (clinical classification)
bash run.sh clinical

Reproduction instructions: Seeds are set inside each script; configs/flags are documented at the top of the scripts. We included task specific datasets in the repo so the demo runs without restrictions.

Code overview / pseudocode
Overall pipeline and figure-generation logic lives in:

Viscosity: src/shared/viscosity/*.py

Clearance: src/shared/clearance/*.py

Clinical: src/shared/clinical/*.py

Each panel script trains/evaluates and writes plots/tables to results/<task>/.



