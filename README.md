# HMM Elderly Activity Monitoring

Human activity recognition from smartphone accelerometer and gyroscope data, using a Hidden Markov Model implemented from scratch in NumPy. The use case is elderly fall-risk and inactivity monitoring. The model infers four hidden activity states (**still, standing, walking, jumping**) from windowed time- and frequency-domain features, using Baum–Welch for training and Viterbi for decoding. Jumping stands in for an impact activity; it is not fall data.

**Report:** [report/HMM_Activity_Recognition_Report.pdf](report/HMM_Activity_Recognition_Report.pdf) (also as `.docx`)
**Notebook:** [notebooks/hmm_activity_recognition.ipynb](notebooks/hmm_activity_recognition.ipynb)

## Results at a glance

| Model | Held-out test windows correct |
|---|---|
| Supervised HMM | 64/65 (98.46%) |
| **Baum–Welch, transition prior fixed (primary)** | **63/65 (96.92%)** |
| Baum–Welch, free transitions (comparison) | 63/65 (96.92%) |
| Independent Gaussian, no transitions (baseline) | 64/65 (98.46%) |

- **Data:** 52 labelled single-activity clips (13 per activity, 5–10 s each, ≥ 90 s per activity), split by whole recording into **44 training and 8 test clips** (11/2 per activity, seed 42). Recorded with Sensor Logger on one Android phone.
- **Sampling:** the phone delivers 58.4 Hz (app set to 100 Hz) or 29.2 Hz (set to 50 Hz), and the accelerometer and gyroscope never share timestamps. Every recording is low-pass filtered at 10 Hz and resampled to a common **25 Hz** grid, so nothing is upsampled.
- **Features:** 2 s windows (50 samples) with a 1 s hop; 10 features (7 time-domain, 3 FFT), Z-scored with training statistics only.
- **Transitions:** every training clip contains one activity, so free Baum–Welch makes still and jumping absorbing (probability 1). The primary model keeps the supervised Laplace transition prior fixed during EM; held-out predictions are identical for both variants.
- **Continuous sessions:** two continuous recordings suffered sensor outages, leaving four unlabelled blocks. They are not used for training, only decoded to show predicted transitions (report Figure 4). The free-A model gets stuck in *jumping*; the primary model switches states but mislabels walking and standing recorded with a different grip.

## Repository structure

```
notebooks/      hmm_activity_recognition.ipynb (the HMM class is defined in Section 5)
scripts/        preprocessing.py, splits.py, features.py, hmm.py, verify_hmm.py,
                train_activity.py, evaluate_activity.py, activity_plots.py,
                build_report.py, redact_metadata.py
data/
  raw/          Sensor Logger exports (pilot, clips, sessions); only the device id is redacted
  processed/    25 Hz labelled CSVs: train/clips (44), test (8); unlabelled/sessions (4 blocks)
  labels/       Approximate session schedule (not verified ground truth)
  metadata.csv  One row per recording (activity, split, sampling rates, duration, date)
  split_manifest.csv  Frozen train/test membership with source hashes
  quality/      Sampling, gap, crop and readiness reports
  features/     Normalised per-window features and the training-only transform
  verification/ Exact and synthetic HMM verification evidence
  models/       Trained model parameters, state mappings and likelihood histories
  evaluation/   Predictions, metrics, confusion matrices and session decoding
figures/        Figures exported from the notebook
report/         Final report (.docx and .pdf) and its configuration
tests/          Numerical, data-integrity and notebook-sync regression tests
```

Processed CSVs use the schema `timestamp_ns, seconds_elapsed, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, activity` (unlabelled blocks omit `activity`). `timestamp_ns` is Unix epoch nanoseconds; `seconds_elapsed` counts from the original recording start; consecutive rows are exactly 40 ms apart.

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/hmm_activity_recognition.ipynb
```

Run all cells from top to bottom (about 80 s). All paths are relative to the repository root, and re-running reproduces every model parameter and result exactly.

To run the pipeline and checks without Jupyter:

```bash
python scripts/preprocessing.py --require-complete
python scripts/features.py
python scripts/verify_hmm.py
python scripts/train_activity.py
python scripts/evaluate_activity.py
python scripts/activity_plots.py
python -m unittest discover -s tests -v
```

To rebuild the report (needs `pip install -r requirements-report.txt`, LibreOffice for the PDF, and a local `videos.txt` with the four video links, which is not committed):

```bash
python scripts/build_report.py
soffice --headless --convert-to pdf --outdir report report/HMM_Activity_Recognition_Report.docx
```

## Scope

This is a single-participant, single-device study. The test clips are held-out recordings from the same collection, so they measure recognition on new recordings rather than in a new environment, and the 65 test windows overlap and are not independent trials. Transition probabilities are a persistence prior rather than learned switching rates, and performance on real falls, other people or other phone placements is untested.
