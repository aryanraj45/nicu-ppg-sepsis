# Non-Contact Neonatal Monitoring with Camera-Based PPG

**How good must a non-contact pulse signal be to support early sepsis warning in the NICU?**

Biomedical Instrumentation project, Department of Instrumentation and Control Engineering, NSUT.
Team: Ishaan Singhal (2024UIC4020), Ansh Jayara (2024UIC4050), Raj Aryan (2024UIC4038). Supervisor: Jyoti Yadav.

`nicu_ppg_sepsis.ipynb` contains the experiments behind the end-semester review. It runs in about 6 min on CPU.

| Part | Experiment | Data |
|---|---|---|
| A | Heart-rate pipeline: FFT → NLMS motion cancelling → spectral tracking → confidence gate | IEEE SPC 2015 (12 recordings, ECG reference) |
| B (E1) | Frame rate and noise vs HRV feature error | PhysioNet PICSDB (10 preterm infants) |
| C (E2) | ML risk model (logistic regression, random forest, leave-one-infant-out): do HeRO-type sepsis patterns survive camera noise? | PICSDB + injected sepsis-like patterns |
| D (E4) | Confidence-gate ablation under motion | PICSDB + motion model |

Research prototype, not a medical device. The sepsis-like patterns in Part C are synthetic; the experiment tests whether the signal survives the camera, not diagnosis.

## Repository layout
```
nicu_ppg_sepsis.ipynb   executed notebook (outputs included)
notebook_src.py         notebook source (# %% cells); edit this, then run build.py
build.py                notebook_src.py -> nicu_ppg_sepsis.ipynb
download_data.sh        fetches both public datasets into ./data
figures/                charts (PNG) and result tables (CSV) for the report and slides
```

## Run locally
```
./download_data.sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/jupyter nbconvert --to notebook --execute --inplace nicu_ppg_sepsis.ipynb
```

## Run on Kaggle
1. Run `./download_data.sh`, zip the `data/` folder, and upload it as a **private** Kaggle dataset (Datasets → New Dataset).
2. Code → New Notebook → File → Import Notebook → upload `nicu_ppg_sepsis.ipynb`.
3. Add Input → select your dataset. Settings → **Internet: On** (the first cell installs `wfdb`).
4. Run All, then Save Version → Save & Run All (Commit).

The notebook finds the data anywhere under `/kaggle/input` or `./data`.

## Data sources
- IEEE Signal Processing Cup 2015 training set. Zhang, Pi & Liu, "TROIKA", IEEE TBME 62(2), 2015.
- PhysioNet PICSDB v1.0.0, Preterm Infant Cardio-Respiratory Signals Database (ODC-By 1.0). https://physionet.org/content/picsdb/1.0.0/
