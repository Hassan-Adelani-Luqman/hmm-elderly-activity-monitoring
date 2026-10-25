# HMM Elderly Activity Monitoring

Human activity recognition from smartphone accelerometer and gyroscope data, using a Hidden Markov Model implemented from scratch in NumPy. The use case is elderly fall-risk and inactivity monitoring. The model infers four hidden activity states (**still, standing, walking, jumping**) from windowed time- and frequency-domain features, using Baum–Welch for training and Viterbi for decoding.

## Repository Structure

```
scripts/        redact_metadata.py (removes the app's device id before data is committed)
data/
  raw/          Sensor Logger exports (pilot, clips, sessions, test); only the device id is redacted
  processed/    Cleaned, 25 Hz resampled, labelled CSVs (train/clips, train/sessions, test)
  labels/       Segment tables for continuous sessions
  metadata.csv  One row per recording (activity, sampling rates, duration, date)
notebooks/      hmm_activity_recognition.ipynb
figures/        Figures exported from the notebook
report/         Final report (.docx and .pdf)
```

## How to Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/hmm_activity_recognition.ipynb
```

Run all cells from top to bottom. All paths are relative to the repository root.

## Report

_Link to be added._
