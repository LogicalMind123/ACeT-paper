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
