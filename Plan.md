# Plan — Formative 2: Modeling Human Activity States with Hidden Markov Models

**Type:** Individual project · **Due:** Sun 1 Nov 2026, 11:59 pm · **Points:** 50
**Use case:** Elderly fall-risk & inactivity monitoring
**Repo:** `hmm-elderly-activity-monitoring` (public GitHub)
**Deliverables:** Public GitHub repo (labelled CSV dataset folder + `.ipynb` HMM notebook) and a 4–5 page report (`.docx`, figures included in the page count) with the repo link on page 1.

The plan is split into **10 phases (0–9)**. Each phase ends with a **⏸ Verification checkpoint**: work stops until the checkpoint passes and you approve moving on. Each phase also lists its **sequential commits**.

---

## Current status — corrected 29 Oct 2026

- **Phase 1:** pilot conclusions verified; jitter wording corrected to ≤0.315 ms for the pilots only.
- **Phase 2:** 52 clips and two incomplete session exports exist. All four links are received in the Git-ignored `videos.txt`. Unauthenticated YouTube metadata and previews are accessible (14–45 s total duration). Preview review found full-body framing issues in all four videos and absent face framing during the still demonstration; compliance remains open.
- **Phase 3:** corrected preprocessing, all-clip validation, metadata, figures and quality reports. The 52 clips retain 5–10 s each and ≥90 s per activity. Session gaps are never interpolated across.
- **Phase 4 complete under revised design:** a frozen whole-recording split assigns 44 clips to training and 8 to testing (11/2 per activity; seed 42). All 52 remain collected data. The test clips were previously quality-inspected but have never been used for learned feature/model fitting. They assess same-collection single-activity recognition, not a new day/place or continuous cross-activity transitions. See `data/split_manifest.csv` and `docs/phase4_test_data.md`.
- **No-re-recording decision:** the participant has no session footage or precise timer records; the approximate 15 s schedule does not verify activity boundaries. Keep original labels only as provisional diagnostic annotations. Recover four independent **unlabelled** session blocks, with original timestamps and recorded orientation. Use the 44 labelled training clips for supervised initialisation and an explicit transition prior; optionally refine with Baum–Welch on unlabelled blocks later. Two labelled training sessions were a design choice and are no longer a completion requirement. Eight whole-recording held-out test clips now supply the evaluation partition.
- **Remaining phase 1–3 evidence:** resolve the video framing findings, verify active phone logging and the matching recordings, and perform a final signed-out playback check. See `docs/video_review.md`. See `docs/collection_completion.md` and `docs/video_checklist.csv`.
- Generated evidence: `data/quality/status.json`, `sampling.csv`, `gaps.csv`, `clip_intervals.csv`, `unlabelled_sequences.csv`, and the executed notebook. `training_ready` and `test_ready` certify the valid collection and fixed 44/8 partition only; it does not certify videos or session ground truth.

The original timeline and commit lists below are historical planning targets, not evidence that a task or checkpoint passed. The revisions in this status section and A1 row 17 supersede the original requirement for labelled training transitions.

---

## Part A — Reference

### A1. Decisions Log (settled 24 Oct 2026)

| # | Topic | Decision |
|---|---|---|
| 1 | Group vs individual | **Individual.** First person; one participant |
| 2 | Dataset size | **52 single-activity clips (13 per activity)**, giving ≥50 files and ≥90 s per activity after reviewed cropping. Four independent unlabelled blocks are recovered from the two incomplete training session exports (row 17) |
| 3 | Devices | **One Android phone** (Infinix X6855, Android 36, Sensor Logger 1.68). Two measured rates exist in the collection/pilots. All selected training and test clips use the 100 Hz setting; the 50 Hz pilot demonstrates lower-rate processing only. Everything is harmonised to **25 Hz** (see rows 15 and 18) |
| 4 | Test data | **8 held-out single-activity clips**, 2 per activity, selected once before feature/model fitting from the existing 52 clips. Shared days/device/participant; no continuous-transition or new-environment evaluation claim (row 18) |
| 5 | Submission | `.docx` upload with the GitHub link on page 1; report copy also committed to the repo |
| 6 | Results table | "Number of Samples" = **test windows**; per-row accuracy = one-vs-rest; an extra **Overall** row gives true accuracy |
| 7 | Still video | Show yourself placing the phone on the table and standing beside it, face visible |
| 8 | Phone OS | **Android**: m/s², rad/s, irregular event-driven timestamps |
| 9 | Use case | **Elderly fall-risk / inactivity monitoring.** Jumping stands in for high-impact events such as stumbles or falls |
| 10 | Verification | **No `hmmlearn`.** Check against exact mathematical ground truth in the notebook |
| 11 | Page limit | 5 pages **including figures**, so a 4-figure budget |
| 12 | Repo | Public. Videos hosted externally, not committed. No location or GPS data |
| 13 | Commits | Incremental conventional commits; **no AI co-author or attribution trailers** |
| 14 | Streams *(pilot, 25 Oct)* | **acc = `TotalAcceleration.csv`** (includes gravity, m/s²); **gyro = `Gyroscope.csv`** (rad/s). The app's `Accelerometer.csv` has gravity removed and is not used. Gravity axis: flat → **z ≈ 9.8**; waist grip → **y ≈ 9.7** (z ≈ 1.3), so still vs standing is separable |
| 15 | Sampling rate *(pilot, 25 Oct)* | App setting 100 Hz → **delivers 58.4 Hz**; setting 50 Hz → **delivers 29.2 Hz**. Pilot jitter σ ≤ 0.315 ms, no dropped intervals; acc and gyro share **no** timestamps. All selected training/test clips use the 100 Hz setting; the 50 Hz pilot is not a held-out test. **Harmonise to 25 Hz**: anti-alias filter + downsampling, never upsampling |
| 16 | Privacy | `device id` in every `Metadata.csv` is redacted with `scripts/redact_metadata.py` before any raw-data commit |
| 17 | Recovery without re-recording *(29 Oct)* | Split sessions at the union of both sensors’ gaps before filtering; discard short blocks, trim 1 s at block ends, preserve recorded orientation and original time. Export surviving blocks **without labels**. Existing segment times are approximate and cannot be independently verified. Labelled clips support supervised emissions; cross-activity transitions initially use an explicit prior. Optional EM treats each recovered block as a separate sequence. |
| 18 | Test partition *(29 Oct)* | User clarified that the existing 52 clips should supply train/test data. Freeze 44 train / 8 test (11/2 per activity), sampled once by whole recording with seed 42. Keep crop/filter settings unchanged; fit features/scaling/models using train only. Disclose earlier quality inspection and limited same-collection single-activity evaluation. |

### A2. Rubric → Phase Map

| Rubric criterion (pts) | "Excellent" requires | Phase |
|---|---|---|
| Data Collection & Quality (10) | ≥50 labelled 5–10 s files; ≥90 s per activity; window and sampling-rate justification; raw-data plots; **4 well-lit videos, face visible** | 1, 2, 3, 4 |
| Feature Extraction (10) | >2 time-domain + >1 FFT features, each justified; Z-score with a reason given | 5 |
| Implementation (15) | Error-free Viterbi; Baum–Welch with **ΔLL < ε** stopping; modular, documented code | 6 |
| Evaluation on Unseen Data (10) | 2 test files; sensitivity, specificity, accuracy; transition + emission plots; test confusion matrix; personalised discussion | 4, 7 |
| Report (5) | 4–5 pages; the brief's headings; captions | 8 |

**Instant-zero risk:** a missing video for any one of the 4 activities scores zero (Phase 2).

### A3. Target Repository Structure

```
hmm-elderly-activity-monitoring/
├── README.md
├── Plan.md
├── requirements.txt
├── .gitignore
├── scripts/
│   └── redact_metadata.py       # removes the device id from Metadata.csv files
├── data/
│   ├── raw/                     # untouched Sensor Logger exports (unzipped CSVs)
│   │   ├── pilot/
│   │   ├── clips/<activity>/<activity>_NN/Accelerometer.csv, Gyroscope.csv, Gravity.csv
│   │   ├── sessions/train_session_0N/
│   │   └── test/test_session_0N/
│   ├── processed/
│   │   ├── train/clips/         # 44 clips: 11 per activity
│   │   ├── train/sessions/      # train_session_01.csv, train_session_02.csv
│   │   └── test/                # 8 held-out single-activity CSVs (split_manifest.csv)
│   ├── labels/                  # session segment tables: start_s, end_s, activity
│   └── metadata.csv             # file, split, activity, configured_hz, measured_acc_hz,
│                                #   measured_gyro_hz, duration_s, date, location_type
├── notebooks/
│   └── hmm_activity_recognition.ipynb
├── figures/
└── report/
    └── HMM_Activity_Recognition_Report.docx  (+ PDF)
```

**Labelled processed CSV schema:** `timestamp_ns, seconds_elapsed, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, activity`.
`timestamp_ns` is integer Unix epoch nanoseconds; `seconds_elapsed` is relative to the original recording start, even after cropping. The two representations share that origin. Plotting may subtract the selected interval start. Unlabelled session files in `data/processed/unlabelled/sessions/` omit `activity`; `data/quality/unlabelled_sequences.csv` records their boundaries and provenance.

### A4. Commit & Checkpoint Rules

- **Format:** [Conventional Commits](https://www.conventionalcommits.org/): `type(scope): imperative summary`, ≤72 characters. Add a body only when the reason isn't obvious.
  - Types: `chore` (setup/tooling), `docs`, `data` (dataset files), `feat` (notebook functionality), `test` (verification checks), `fix`, `refactor`.
- **No `Co-Authored-By` lines or any AI attribution** in commit messages or PR descriptions. Commits are authored by your own git identity only.
- **One logical unit per commit**, made as each task completes. Notebook commits follow the notebook's section order, so the history reads like the notebook being built.
- **The checkpoint gates the push:**
  1. Finish the phase's tasks and commit them locally in the listed order.
  2. ⏸ **Pause.** Review the checkpoint list and `git log --oneline`.
  3. If anything fails, add a `fix:` commit. Don't rewrite pushed history.
  4. When you approve, `git push` and start the next phase.
- Notebook outputs are **kept** (the grader needs to see the plots). Before committing a notebook change, re-run the affected sections so the outputs match the code.
- **Never committed:** videos, zip exports, `.ipynb_checkpoints/`, `.venv/`, `__pycache__/`, Office lock files (`~$*.docx`), location data.

### A5. Timeline

| Phase | Date | Theme |
|---|---|---|
| 0 | Sat 24 Oct | Repository setup |
| 1 | Sun 25 Oct (morning) | App setup & pilot recording |
| 2 | Sun 25 Oct | Training data collection & videos |
| 3 | Mon 26 Oct | Pre-processing & harmonisation |
| 4 | Tue 27 Oct (morning) | Unseen test data collection |
| 5 | Tue 27 Oct | Windowing, features, normalisation |
| 6 | Wed 28 Oct | HMM implementation & verification |
| 7 | Thu 29 Oct | Training, evaluation & visualisation |
| 8 | Fri 30 – Sat 31 Oct | Report & README |
| 9 | Sat 31 Oct – Sun 1 Nov | Final QA & submission |

---

## Part B — Phases

## Phase 0 — Repository Setup
**Goal:** an empty but well-structured repo, pushed to GitHub.

**Tasks**
- [ ] Check `git config user.name` and `user.email` (commits must show your identity).
- [ ] `git init` on branch `main`.
- [ ] Create the folders from A3, using `.gitkeep` in empty ones.
- [ ] `.gitignore` covering everything in A4's "Never committed" list, plus `*.mp4 *.mov *.zip`.
- [ ] `requirements.txt` with pinned versions: numpy, pandas, scipy, matplotlib, seaborn, jupyter.
- [ ] Set up a virtual environment and `pip install -r requirements.txt`; check that `jupyter` starts.
- [ ] `README.md` skeleton: title, use case (1 line), structure, how to run, report link (placeholder).
- [ ] Create the **public** GitHub repo `hmm-elderly-activity-monitoring` and add it as `origin`.

**Commits**
1. `docs: add project plan`
2. `chore: scaffold repository directory structure`
3. `chore: add gitignore and pinned requirements`
4. `docs: add README skeleton`

**⏸ Checkpoint 0**
- [ ] `git log --oneline` shows the 4 commits with your name and no co-author trailer.
- [ ] `git status` is clean; the venv is not tracked.
- [ ] The repo is visible on GitHub and public.

---

## Phase 1 — App Setup & Pilot Recording
**Goal:** confirm the data format before recording 52 clips.

**Tasks**
- [x] Install **Sensor Logger** (Android). Enable **Accelerometer, Gyroscope, Gravity** (and Total Acceleration if listed). **Turn off Location/GPS and all other sensors.**
- [x] Set the sampling rate to 100 Hz and note the **configured** rate.
- [x] Record 4 pilots (flat, upright grip, walk, flat at the 50 Hz setting), export as CSV, unzip into `data/raw/pilot/`.
- [x] Redact the device id (`scripts/redact_metadata.py`).
- [x] Start the notebook with **Section 1 "Setup & data loading"**: imports, path constants, `load_recording()`, measured-rate helper, pilot check.
- [x] Inspect the pilots: timestamp format, gravity, gravity axis per orientation, measured rates and jitter, rate adjustability.
- [x] Write the answers into Plan.md A1 (rows 14–16).

**Commits**
1. `chore: add metadata redaction script`
2. `data: add pilot recordings`
3. `feat(notebook): add setup and sensor loading utilities`
4. `docs: record pilot findings in plan and README`

**⏸ Checkpoint 1**
- [x] The pilots load with no errors; units are m/s² and rad/s.
- [x] Gravity question answered; feature 1–3 source decided (`TotalAcceleration` axis means).
- [x] Measured rates noted; sampling-rate adjustability known (100 Hz → 58.4 Hz; 50 Hz → 29.2 Hz).
- [x] No location files, and no device id in any committed `Metadata.csv`.
- [x] **App set back to the 100 Hz setting** for Phase 2.

---

## Phase 2 — Training Data Collection & Videos
**Goal:** all training recordings and the 4 mandatory videos. Links, metadata duration and frame previews have been checked. Full-body framing has not passed; full playback and active logging remain unverified. The revised session recovery strategy is in A1 row 17.

**Recording protocol**

| Activity | Phone placement | Instructions |
|---|---|---|
| **Standing** | Held in hand, screen facing body, waist level | Stand naturally, no swaying |
| **Walking** | Same grip and position | Steady normal pace, flat ground |
| **Jumping** | Same grip and position | Continuous two-footed jumps, steady rhythm (stand-in for impact events) |
| **Still** | Flat on a table, screen up | Hands off; nobody touches the table |

**Tasks**
- [ ] **Video compliance needs attention** (links received; full-body framing fails in reviewed previews; missing any one at submission scores zero):
  - 4 videos, ≥5 s each, one per activity.
  - **Entire body in frame, face clearly visible, front lighting.**
  - Use the **laptop webcam** 2–3 m away, or a friend with any camera.
  - Still video: show yourself placing the phone on the table and standing beside it.
  - Film each video **during a real clip** and note the matching file (e.g. "Video 2 ↔ `walking_03`").
- [x] **52 clips:** 13 per activity; reviewed crops now span 5–10 s and exceed 90 s per activity in total.
  - Batch by activity, in 2 sittings, with slight variation in context (room, time, footwear).
  - **Never change the phone placement.**
- [x] **Two raw training sessions retained for provenance; recover continuous portions without labels.**
  - Original exports have sensor outages; use the four independent recovered blocks only as optional unlabelled sequences.
  - The old segment table remains approximate and is excluded from supervised transitions and accuracy metrics.
  - Exact timing evidence is unavailable; no boundary is invented or certified from the rough 15-second schedule.
- [x] Check that the app is on the **100 Hz setting** (delivers 58.4 Hz) before every sitting.
- [x] Unzip every export into the `data/raw/...` layout from A3, then run `python scripts/redact_metadata.py data/raw` before committing.
- [x] Fill in `data/metadata.csv` for every recording.
- [x] Four videos hosted on YouTube; links received in Git-ignored `videos.txt` for the report. Unauthenticated metadata and preview access succeeded; full playback and rubric compliance remain separate checks.

**Commits** (one per activity keeps the history readable)
1. `data: add raw standing clips`
2. `data: add raw walking clips`
3. `data: add raw jumping clips`
4. `data: add raw still clips`
5. `data: add raw training transition sessions and segment labels`
6. `data: add recording metadata`

**⏸ Checkpoint 2**
- [x] 52 clip folders present (13 per activity), each with Accelerometer, Gyroscope and Gravity CSVs.
- [x] Two raw session folders and the original approximate annotations retained; four independently processed unlabelled blocks exported under the revised strategy. No verified session ground truth is claimed.
- [x] `metadata.csv` has one row per recording.
- [ ] **All 4 video links play in an incognito window**: full body, face visible, well lit.
- [x] No video, zip, or location file is tracked (`git ls-files | grep -Ei 'mp4|mov|zip|location'` returns nothing).

---

## Phase 3 — Pre-processing & Harmonisation
**Goal:** clean, labelled, uniformly sampled training CSVs, plus the raw-data figure.

**Tasks** (notebook Section 2 "Pre-processing")
- [x] **Quality checks:** measure every sensor/file; report all gaps (Δt > 3 × median), jitter and duplicate removal; reject conflicting duplicates/non-finite data. Split sessions at either sensor’s gaps before filtering; treat blocks independently.
- [x] **Harmonisation:**
  - Why it's needed (from the pilot): recordings exist at **two real rates** (58.4 Hz and 29.2 Hz, from the 100 and 50 Hz settings), and accelerometer and gyroscope samples are taken at **different instants** (no shared timestamps).
  - Step 1: an **anti-alias low-pass filter** at each stream's native rate: zero-phase Butterworth (`scipy.signal.butter` + `sosfiltfilt`, second-order sections), order 4, **cutoff 10 Hz** (below the new 12.5 Hz Nyquist).
  - Step 2: build a **uniform 25 Hz grid** over the time range both sensors cover, and linearly interpolate all six filtered native axes directly onto it **once** (`np.interp`). Use integer nanosecond steps to prevent rounding drift.
  - Why 25 Hz: it is just below the lowest delivered rate (29.2 Hz), so every recording is **downsampled and nothing is upsampled**. Interpolating up would invent no content above the source Nyquist, and test FFT features would no longer match training. The 12.5 Hz Nyquist still covers the walking and jumping fundamentals (~1.5–2.5 Hz) and their main harmonics. Window sizes in samples become identical for all recordings.
- [x] **Cleaning:** exclude at least 1 s at both ends, apply signal-reviewed crop intervals bound to raw-file hashes, cap at 10 s and enforce 5–10 s on every clip. Further remove obvious quiet lead-ins/handling artifacts. No re-recording was needed for the 52 clips.
- [x] **Labelling:** retain original clip activity labels. Export recovered session blocks **without** labels; approximate session timings are not ground truth. Strict label validation exists for future independently verified sessions. No inferred labels enter training.
- [x] Save labelled clips to `data/processed/train/clips/`; unlabelled blocks to `data/processed/unlabelled/sessions/`. Remove obsolete unsafe derived session CSVs; retain raw data.
- [x] **Figures:**
  - **Fig. 1 (report):** 4 panels of actual raw measurements over the retained intervals, one clip per activity, |a| and |ω| with separate labelled scales.
  - Notebook only: all 6 axes per activity; Δt histogram before and after resampling; seconds-per-activity bar chart with a 90 s line.
- [x] Sampling-rate summary table (configured vs measured vs harmonised) for the report.

**Commits**
1. `feat(notebook): add sampling quality checks`
2. `feat(notebook): add 25 Hz resampling with anti-alias filter`
3. `feat(notebook): add clip trimming and session labelling`
4. `data: add processed training clips and sessions`
5. `feat(notebook): add raw signal and sampling visualisations`

**⏸ Checkpoint 3**
- [x] 52 processed clips, each 5–10 s, with no NaNs and Δt = 0.04 s (25 Hz) throughout.
- [x] Every activity totals ≥90 s (bar chart).
- [x] Under the revised strategy, session labels are not asserted as truth. Raw diagnostics and the recoverable-block manifest document the uncertainty; no gaps are crossed and no labels are exported with those blocks.
- [x] Fig. 1 is readable at report size and saved to `figures/`.
- [x] The notebook runs top-to-bottom up to this point.

---

## Phase 4 — Held-Out Test Partition
**Goal:** a fixed, disjoint evaluation subset of the existing independent recordings, selected before feature extraction or model fitting.

**Revised decision (29 Oct)**
The participant intended some of the 52 clips for testing. Use a recording-level
holdout: 44 train and 8 test, with 11/2 recordings per activity. This supersedes
the original two new continuous sessions, different-day/place and lower-rate test
requirements, which were extra design choices. The rubric's test evaluation is
now supplied by eight held-out single-activity files. Report the narrower scope
and the fact that all clips had already undergone quality inspection/cropping.

**Tasks**
- [x] Select two clips per activity once using seed 42; freeze membership and raw hashes in `data/split_manifest.csv`.
- [x] Keep all original recordings intact and preserve their labels/timestamps.
- [x] Save 44 clips under `data/processed/train/clips/` and 8 under `data/processed/test/`; remove held-out copies from training outputs.
- [x] Process both splits with the unchanged 10 Hz / 25 Hz Phase 3 functions and frozen crop intervals.
- [x] Update metadata, quality summaries and notebook with actual split, dates, rates and limitations.
- [x] Add source/provenance, duplicate-recording, output membership and sequence-separation checks.
- [x] Provide explicit train/test sequence loaders; all learned fitting/tuning is restricted to train in subsequent phases.

**Checkpoint 4**
- [x] 44 train and 8 test files; exactly 11/2 per activity, all four activities represented.
- [x] No duplicate original recording crosses the split; unlabelled training blocks do not overlap held-out acquisition times.
- [x] Each test file has its original single-activity label, finite data, 5–10 s duration and exact 25 Hz spacing.
- [x] The 52-file/90-second requirements are checked on the collected dataset, not claimed for training alone.
- [x] No model/learned preprocessing was fitted before the split; previous quality inspection is disclosed. No evaluation results have been used for tuning.
- [x] The notebook runs top-to-bottom and the preprocessing/split regression checks pass.

**Scope limits:** existing single-person/device/indoor collection, shared collection
days, no unseen lower-rate test, and no real cross-activity test transitions.
Do not concatenate clips into artificial sequences. Video framing compliance
remains open independently. See `docs/phase4_test_data.md`.

---

## Phase 5 — Windowing, Feature Extraction & Normalisation
**Goal:** separate Z-scored feature sequences for the 44 training and 8 held-out test clips (plus optional unlabelled training blocks).

**Windowing:** **2.0 s = 50 samples at 25 Hz, hop 1.0 s.** The window is defined in seconds and converted with `fs × seconds`.
- Walking runs at ~1.6–2 steps/s and jumping at ~1.5–2.5 jumps/s, so a window covers ≥3 cycles.
- FFT resolution 1/T = 0.5 Hz.
- Compute actual training/test window counts from the frozen split; keep each recording as a separate sequence.
- A window takes the majority label of its samples.

**Features (~12–14 per window):**

| # | Feature | Domain | Why it helps |
|---|---|---|---|
| 1–3 | Mean of each gravity / acc axis | Time | Phone orientation: flat (still) puts g on z; upright (standing) puts g on y. **Main still-vs-standing cue** |
| 4 | Std of \|a\| | Time | Motion intensity: still < standing < walking < jumping |
| 5 | Variance of \|ω\| | Time | Hand rotation during walking and jumping |
| 6 | RMS of (\|a\| − mean) | Time | Dynamic energy; jump impacts dominate |
| 7–8 | SMA (acc), SMA (gyro) | Time | Standard HAR measure of activity level |
| 9 | Peak-to-peak \|a\| | Time | Landing spikes, the fall-like impact signature |
| 10 | Corr(acc_y, acc_z) | Time | Coordinated movement vs uncorrelated noise |
| 11 | Dominant frequency of \|a\| (FFT) | Freq | Step or jump rate; no clear peak when still or standing |
| 12 | Spectral energy of \|a\| | Freq | Periodic high-energy motion vs near-silence |
| 13 | Band-energy ratio (0.5–3 / 3–10 Hz) | Freq | Gait energy is low-band; impacts and noise are high-band (upper band capped by the 10 Hz anti-alias cutoff) |
| 14 | Spectral entropy | Freq | Periodic motion has low entropy; noise has high entropy |

FFT details: remove the mean, apply a Hann window, use `np.fft.rfft`, and skip the DC bin when finding the peak.

**Normalisation: Z-score, fitted on training windows only**, then applied to the test windows. Reasons:
- The features have mixed units and scales.
- Diagonal Gaussian emissions would otherwise be dominated by large-scale features.
- Min-max scaling is distorted by jump spikes.
- Fitting on train only avoids leakage.

**Tasks** (notebook Section 3 "Features")
- [ ] `window_signal()` with majority labels.
- [ ] Time-domain features (1–10).
- [ ] FFT features (11–14).
- [ ] `zscore_fit()` / `zscore_transform()`.
- [ ] Training-only per-class box plots and a feature-correlation heatmap; drop any feature with |r| > 0.95 against another (keep D ≤ ~14).

**Commits**
1. `feat(notebook): add sliding-window segmentation`
2. `feat(notebook): add time-domain features`
3. `feat(notebook): add FFT frequency-domain features`
4. `feat(notebook): add z-score normalisation fitted on training data`
5. `feat(notebook): add feature distribution and correlation plots`

**⏸ Checkpoint 5**
- [ ] The feature matrix has no NaN or inf values. Training features have mean ≈ 0 and std ≈ 1; test features are transformed with the **training** statistics.
- [ ] The box plots show the expected ordering (e.g. std|a|: still < standing < walking < jumping).
- [ ] Still vs standing is separable on at least one orientation feature.
- [ ] The final feature list is recorded in the notebook with one-line justifications.

---

## Phase 6 — HMM Implementation & Verification
**Goal:** a correct, from-scratch numpy `GaussianHMM`, with verification shown in the notebook.

**Model components**

| Element | Definition |
|---|---|
| **Z** | N = 4: {0 Still, 1 Standing, 2 Walking, 3 Jumping} |
| **X** | Z-scored feature vector x_t ∈ ℝᴰ per window |
| **A** | A[i,j] = P(z_t = j \| z_{t−1} = i) |
| **B** | Diagonal Gaussian per state, 𝒩(x; μ_j, diag σ²_j), variance floor 1e-3 |
| **π** | π_i = P(z_1 = i) |

- **Why diagonal:** with ~175 windows per state, a full covariance (~100 parameters per state) would be poorly estimated. The variance floor stops the still state's variances from collapsing to zero.
- **Algorithms** (brief in the report, tied to the metrics):
  - **Forward** gives P(X | λ), the convergence criterion.
  - **Baum–Welch** (EM via forward–backward) learns λ.
  - **Viterbi** gives the most probable state path. The metrics are computed on per-window hard labels, and Viterbi uses A to keep those labels temporally consistent.

**Class design**
```python
class GaussianHMM:
    """Diagonal-covariance Gaussian HMM implemented from scratch in numpy."""
    def __init__(self, n_states, n_features, var_floor=1e-3, tol=1e-4, max_iter=500): ...
    def fit_supervised(self, X_list, y_list): ...   # counts + Laplace smoothing
    def _log_emission(self, X): ...                 # (T, N)
    def _forward(self, log_B): ...                  # scaled alphas, scales, log-likelihood
    def _backward(self, log_B, scales): ...
    def score(self, X_list): ...
    def fit_baum_welch(self, X_list): ...           # stop when |ΔLL| < tol; max_iter = safety cap
    def predict_viterbi(self, X): ...               # log space with backpointers
    def predict_proba(self, X): ...                 # posteriors γ
```

**Verification** (notebook Section 6 "Model verification", all as assertions). Why: Implementation is worth 15 points; from-scratch HMM bugs are usually silent; and exact ground truth is stronger evidence than agreeing with a library.
1. **Brute-force likelihood:** for N = 2, T = 6 (64 paths), summing over every path must equal `exp(score)`.
2. **Brute-force Viterbi:** the argmax over all 64 paths must equal the Viterbi path.
3. **Forward–backward consistency:** P(X) from α must equal P(X) from β; each γ_t and ξ_t must sum to 1.
4. **EM monotonicity:** the LL must never decrease.
5. **Synthetic recovery:** sample from a known 3-state HMM, fit from a random start, match states, and check A and μ are within tolerance.
6. **Invariants:** rows of A and π sum to 1, and every variance is ≥ the floor, after every M-step.

**Tasks** (notebook Section 5 "HMM implementation" + Section 6)
- [ ] Emissions + scaled forward/backward + `score`.
- [ ] Viterbi.
- [ ] Baum–Welch for multiple sequences, with ΔLL < 1e-4 stopping and LL history.
- [ ] Supervised initialisation.
- [ ] Verification checks 1–6.

**Commits**
1. `feat(hmm): add Gaussian emissions and scaled forward-backward`
2. `feat(hmm): add Viterbi decoding`
3. `feat(hmm): add Baum-Welch training with log-likelihood convergence`
4. `feat(hmm): add supervised parameter initialisation`
5. `test(hmm): add brute-force and consistency checks`
6. `test(hmm): add synthetic parameter recovery check`

**⏸ Checkpoint 6**
- [ ] All 6 verification checks pass.
- [ ] Baum–Welch stops on ΔLL < ε (not the iteration cap) on the synthetic data.
- [ ] Every method has a docstring with shapes; no real activity data is used yet.

---

## Phase 7 — Training, Evaluation & Visualisation
**Goal:** a trained model evaluated on the unseen test sessions, plus every figure for the results section.

**Training strategy**
1. Supervised emission initialisation from the 44 labelled training clips only. Clips supply persistence observations; cross-activity transitions use an explicit nonzero prior, because session labels are unverified. Do not count cross-transitions from approximate labels.
2. Baum–Welch refinement over training sequences, optionally including recovered **unlabelled** session blocks. Keep every clip/block as its own sequence and never count transitions across file or gap boundaries. Fit preprocessing statistics using training data only; document the session grip difference.
3. State-to-label mapping check (Hungarian on the training confusion matrix).
4. Compare: **(a)** supervised only · **(b)** supervised + Baum–Welch · **(c)** per-window Gaussian classifier with no transitions (baseline). Baum–Welch maximises likelihood, not accuracy, so report whichever result happens and explain it.

**Evaluation**
- Decode all eight held-out clips independently with Viterbi and pool the windows into a 4×4 confusion matrix (per-clip numbers go in the notebook).
- Sensitivity = TP/(TP+FN); Specificity = TN/(TN+FP); per-row accuracy = (TP+TN)/N. Implement these in numpy.

| State (Activity) | Number of Samples (windows) | Sensitivity | Specificity | Overall Accuracy |
|---|---|---|---|---|
| Still | | | | |
| Standing | | | | |
| Walking | | | | |
| Jumping | | | | |
| **Overall** | total | macro avg | macro avg | correct / total |

  The footnote defines each formula and notes that one-vs-rest accuracy is inflated by true negatives, which is why the Overall row and the confusion matrix are the headline results.
- Generalisation: compare training accuracy with test accuracy; report per-clip errors and discuss shared collection conditions. True test cross-activity transitions, a new day/place and a lower-rate unseen recording are not evaluated by this split.

**Report figures** (multi-panel, exported at 150–200 dpi to `figures/`)
- **Fig. 2:** transition-matrix heatmap (annotated) + emission-means heatmap (states × features).
- **Fig. 3:** test confusion matrix + Baum–Welch LL convergence curve.
- **Fig. 4:** separate representative held-out clip timelines (true vs predicted above |a|). Never join independent clips to simulate transitions.
- Notebook only: A before vs after Baum–Welch; Gaussian curves per state for 2–3 key features.

**Tasks** (notebook Section 7 "Training" + Section 8 "Evaluation")
- [ ] Train models (a), (b), (c); mapping check.
- [ ] Metrics functions + results table.
- [ ] Figs. 2–4 + the notebook-only plots.
- [ ] Short interpretation markdown under each result.

**Commits**
1. `feat(notebook): train supervised and Baum-Welch HMMs on activity data`
2. `feat(notebook): add no-transition baseline for comparison`
3. `feat(notebook): add per-class metrics and confusion matrix on test data`
4. `feat(notebook): add transition and emission probability plots`
5. `feat(notebook): add decoded sequence plots for test sessions`
6. `docs(figures): export report figures`

**⏸ Checkpoint 7**
- [ ] The test data was used **only** for final evaluation (no tuning on it).
- [ ] The metrics table is complete and consistent with the confusion matrix (the counts add up).
- [ ] The rows of the transition heatmap sum to 1; the decoded timelines are readable.
- [ ] Figs. 2–4 are legible at half-page width.
- [ ] Restart & Run All succeeds on the whole notebook.

---

## Phase 8 — Report & README
**Goal:** a ≤5-page report (figures included) using the brief's exact headings.

| Section | Pages | Contents |
|---|---|---|
| Title block + **Background and Motivation** | 0.4 | Name, course, date, **repo link**; one paragraph on elderly fall-risk and inactivity monitoring, and why HAR is needed |
| **Data Collection and Pre-processing** | 1.3 | Protocol; **4 video links** labelled by activity; sampling-rate table; harmonisation; cleaning; window justification; feature table + Z-score reasoning; Fig. 1 |
| **HMM Setup and Implementation Details** | 0.9 | Z/X/A/B/π table; algorithm choice (brief, tied to metrics); training strategy; convergence; verification summary |
| **Results and Interpretation** | 1.5 | How the unseen data was obtained; metrics table; Figs. 2–4 with interpretation |
| **Discussion and Conclusion** | 0.9 | See the points below |

**Discussion points**
- **Hardest pair:** likely still vs standing (orientation features separate them). **Easiest:** jumping (very large energy and peak-to-peak values).
- **Transitions:** a strong diagonal shows that activities persist. Still ↔ standing is the gateway (you pick the phone up or put it down while standing); still → jumping should be near zero. In the use case, impact → long still would be an alert pattern.
- **Noise and sampling:** the delivered rate differed from the configured rate (58.4 vs 100 Hz); acc and gyro are sampled at different instants; gyroscope drift; interpolation; aliasing; why 25 Hz is enough and what is lost above 12.5 Hz (sharpness of impacts); the lower-rate pilot demonstrates processing compatibility, but no unseen lower-rate generalisation was tested.
- **Improvements:** more participants and devices; pocket or belt placement (more realistic for older adults); barometer for fall detection; GMM emissions; a duration-aware HSMM.

**Tasks**
- [ ] Draft the report in Word with numbered captions for every figure and table.
- [ ] Edit down to ≤5 pages and spell-check.
- [ ] Export a PDF copy.
- [ ] Complete the README: summary, dataset description, how to run, link to the report. Leave out the video links (privacy); the report carries them.

**Commits**
1. `docs(report): add report draft`
2. `docs(report): finalise report and add PDF export`
3. `docs: complete README with dataset and usage details`

**⏸ Checkpoint 8**
- [ ] ≤5 pages including figures; the headings match the brief exactly.
- [ ] The repo link and the 4 video links in the report all open in incognito.
- [ ] Every number in the report matches the notebook output.

---

## Phase 9 — Final QA & Submission
**Goal:** a clean, reproducible repo and the submitted `.docx`.

**Tasks**
- [ ] Fresh-clone test: clone into a temporary folder, create a new venv, install requirements, Restart & Run All. Everything passes using relative paths only.
- [ ] Run through the final checklist below.
- [ ] Tag the release `v1.0`.
- [ ] Upload the `.docx` to the course portal **before Sun 1 Nov, 11:59 pm**.

**Commits**
1. `fix: …` (only if the fresh-clone test finds issues)
2. `chore: re-run notebook with final outputs`
3. Tag: `git tag -a v1.0 -m "Formative 2 submission"`

**⏸ Checkpoint 9 — Final Submission Checklist**
- [ ] ≥50 labelled clips (52 recorded; 5–10 s, timestamp column); ≥90 s per activity. Optional unlabelled recovered blocks are independent sequences, not verified session labels
- [ ] 8 held-out original clips (2 per activity), fixed whole-recording split, original labels, and same-collection/quality-inspection limitations disclosed
- [ ] 4 videos: full body, face visible, well lit; links tested in incognito
- [ ] Sampling-rate table, harmonisation explanation, window justification
- [ ] ≥3 time-domain + ≥1 FFT features, each justified; Z-score fitted on train only
- [ ] Viterbi + Baum–Welch with ΔLL < ε; verification section passes
- [ ] Transition heatmap, emission plot, confusion matrix, decoded timelines, LL curve
- [ ] Metrics table with footnoted definitions + Overall row
- [ ] Report ≤5 pages including figures, the brief's headings, captions, repo and video links
- [ ] Notebook passes Restart & Run All from a fresh clone; public repo pushed; no AI co-author trailers in `git log`
- [ ] `.docx` uploaded

---

## Part C — Risks & Mitigations

| Risk | Mitigation | Phase |
|---|---|---|
| Still vs standing confusion | Gravity-orientation features; Gravity stream enabled | 1, 5 |
| A few clips fail cleaning, dropping the count below 50 | 52 recorded (buffer of 2); re-record rejected clips straight away | 3 |
| Limited labelled transitions | Explicit persistence/exit prior; optional EM over independent unlabelled blocks; report the limitation | 2, 6 |
| Unverified training-session boundaries | Never use approximate labels as truth; recovered blocks are unlabelled. Record exact evidence for future test data | 2, 4 |
| Underflow / singular variance | Scaled forward–backward; variance floor | 6 |
| Baum–Welch label switching | Supervised initialisation + Hungarian check | 7 |
| Leakage | Separate test day and place; Z-score fitted on train only; no tuning on test | 4, 5, 7 |
| Video links inaccessible | Incognito test; "anyone with link" sharing | 2, 8 |
| Privacy (public repo) | No GPS; videos not committed; `metadata.csv` says "indoor/outdoor", not addresses | 1, 2 |
| Over the page limit | 4-figure budget; extra plots stay in the notebook | 8 |
| Notebook fails for the grader | Relative paths; fresh-clone test; pinned requirements | 9 |
