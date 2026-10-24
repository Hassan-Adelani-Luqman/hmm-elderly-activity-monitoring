# Plan — Formative 2: Modeling Human Activity States with Hidden Markov Models

**Type:** Individual project · **Due:** Sun 1 Nov 2026, 11:59 pm · **Points:** 50
**Use case:** Elderly fall-risk & inactivity monitoring
**Repo:** `hmm-elderly-activity-monitoring` (public GitHub)
**Deliverables:** Public GitHub repo (labelled CSV dataset folder + `.ipynb` HMM notebook) and a 4–5 page report (`.docx`, figures included in the page count) with the repo link on page 1.

The plan is split into **10 phases (0–9)**. Each phase ends with a **⏸ Verification checkpoint**: work stops until the checkpoint passes and you approve moving on. Each phase also lists its **sequential commits**.

---

## Part A — Reference

### A1. Decisions Log (settled 24 Oct 2026)

| # | Topic | Decision |
|---|---|---|
| 1 | Group vs individual | **Individual.** First person; one participant |
| 2 | Dataset size | **52 single-activity clips (13 per activity)**, giving ≥50 files and ~130 s per activity. Plus 2 continuous training sessions so the model sees transitions |
| 3 | Devices | **One Android phone.** Harmonise accelerometer vs gyroscope streams to 50 Hz. If the app allows it, record test session 2 at a different configured rate |
| 4 | Test data | **2 continuous multi-activity sessions**, recorded on a different day and in a different place |
| 5 | Submission | `.docx` upload with the GitHub link on page 1; report copy also committed to the repo |
| 6 | Results table | "Number of Samples" = **test windows**; per-row accuracy = one-vs-rest; an extra **Overall** row gives true accuracy |
| 7 | Still video | Show yourself placing the phone on the table and standing beside it, face visible |
| 8 | Phone OS | **Android**: m/s², rad/s, irregular event-driven timestamps |
| 9 | Use case | **Elderly fall-risk / inactivity monitoring.** Jumping stands in for high-impact events such as stumbles or falls |
| 10 | Verification | **No `hmmlearn`.** Check against exact mathematical ground truth in the notebook |
| 11 | Page limit | 5 pages **including figures**, so a 4-figure budget |
| 12 | Repo | Public. Videos hosted externally, not committed. No location or GPS data |
| 13 | Commits | Incremental conventional commits; **no AI co-author or attribution trailers** |

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
├── data/
│   ├── raw/                     # untouched Sensor Logger exports (unzipped CSVs)
│   │   ├── pilot/
│   │   ├── clips/<activity>/<activity>_NN/Accelerometer.csv, Gyroscope.csv, Gravity.csv
│   │   ├── sessions/train_session_0N/
│   │   └── test/test_session_0N/
│   ├── processed/
│   │   ├── train/clips/         # standing_01.csv … still_13.csv (52 files)
│   │   ├── train/sessions/      # train_session_01.csv, train_session_02.csv
│   │   └── test/                # test_session_01.csv, test_session_02.csv
│   ├── labels/                  # session segment tables: start_s, end_s, activity
│   └── metadata.csv             # file, split, activity, configured_hz, measured_acc_hz,
│                                #   measured_gyro_hz, duration_s, date, location_type
├── notebooks/
│   └── hmm_activity_recognition.ipynb
├── figures/
└── report/
    └── HMM_Activity_Recognition_Report.docx  (+ PDF)
```

**Processed CSV schema:** `timestamp_ns, seconds_elapsed, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, activity`

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
- [ ] Install **Sensor Logger** (Android). Enable **Accelerometer, Gyroscope, Gravity** (and Total Acceleration if listed). **Turn off Location/GPS and all other sensors.**
- [ ] Set the highest stable sampling rate and note the **configured** rate.
- [ ] Record **10 s with the phone flat on a table**, export as CSV, unzip into `data/raw/pilot/`.
- [ ] Start the notebook with **Section 1 "Setup & data loading"**: imports, path constants, `load_recording()`, measured-rate helper.
- [ ] Inspect the pilot and answer these questions:
  - Columns and timestamp format (expect `time` in ns plus `seconds_elapsed`).
  - **Gravity in accelerometer?** Magnitude at rest ≈ 9.81 means included; ≈ 0 means removed, so use the Gravity stream for the orientation features.
  - Measured rate per sensor (`1/median Δt`) and timestamp jitter.
  - Can the app's sampling rate be changed? (This decides the different-rate demo in Phase 4.)
- [ ] Write the answers into Plan.md A1 (new rows 14–15).

**Commits**
1. `data: add pilot recording`
2. `feat(notebook): add setup and sensor loading utilities`
3. `docs: record pilot findings in plan`

**⏸ Checkpoint 1**
- [ ] The pilot loads with no errors; units are m/s² and rad/s.
- [ ] Gravity question answered; feature 1–3 source decided (accelerometer or Gravity stream).
- [ ] Measured rates noted; sampling-rate adjustability known.
- [ ] No location columns or files in the export.

---

## Phase 2 — Training Data Collection & Videos
**Goal:** all training recordings and the 4 mandatory videos.

**Recording protocol**

| Activity | Phone placement | Instructions |
|---|---|---|
| **Standing** | Held in hand, screen facing body, waist level | Stand naturally, no swaying |
| **Walking** | Same grip and position | Steady normal pace, flat ground |
| **Jumping** | Same grip and position | Continuous two-footed jumps, steady rhythm (stand-in for impact events) |
| **Still** | Flat on a table, screen up | Hands off; nobody touches the table |

**Tasks**
- [ ] **Videos first** (mandatory; missing any one scores zero):
  - 4 videos, ≥5 s each, one per activity.
  - **Entire body in frame, face clearly visible, front lighting.**
  - Use the **laptop webcam** 2–3 m away, or a friend with any camera.
  - Still video: show yourself placing the phone on the table and standing beside it.
  - Film each video **during a real clip** and note the matching file (e.g. "Video 2 ↔ `walking_03`").
- [ ] **52 clips:** 13 per activity, each recorded for ~12 s (trimmed to ~10 s in Phase 3).
  - Batch by activity, in 2 sittings, with slight variation in context (room, time, footwear).
  - **Never change the phone placement.**
- [ ] **2 training sessions**, ~2 min each, for transitions:
  - Each activity appears twice for ~15 s, in a realistic order, e.g. `still → standing → walking → standing → jumping → standing → walking → still`.
  - Start with **3 sharp taps** on the phone (sync marker).
  - Follow a **laptop interval timer** with audio cues, and film with the webcam to get exact boundaries.
  - Write the segment table (`start_s, end_s, activity`) right after each session.
- [ ] Unzip every export into the `data/raw/...` layout from A3.
- [ ] Fill in `data/metadata.csv` for every recording.
- [ ] Upload the videos to **YouTube (unlisted)** or **Google Drive (Anyone with link → Viewer)**. Keep the links in a private note for the report; they are not committed.

**Commits** (one per activity keeps the history readable)
1. `data: add raw standing clips`
2. `data: add raw walking clips`
3. `data: add raw jumping clips`
4. `data: add raw still clips`
5. `data: add raw training transition sessions and segment labels`
6. `data: add recording metadata`

**⏸ Checkpoint 2**
- [ ] 52 clip folders present (13 per activity), each with Accelerometer, Gyroscope and Gravity CSVs.
- [ ] 2 session folders plus their segment tables.
- [ ] `metadata.csv` has one row per recording.
- [ ] **All 4 video links play in an incognito window**: full body, face visible, well lit.
- [ ] No video, zip, or location file is tracked (`git ls-files | grep -Ei 'mp4|mov|zip|location'` returns nothing).

---

## Phase 3 — Pre-processing & Harmonisation
**Goal:** clean, labelled, uniformly sampled training CSVs, plus the raw-data figure.

**Tasks** (notebook Section 2 "Pre-processing")
- [ ] **Quality checks:** measured rate per sensor and file; flag gaps (Δt > 3 × median); drop duplicate timestamps.
- [ ] **Harmonisation:**
  - Android sensors are event-driven, so the accelerometer and gyroscope run at different, jittery rates on separate timestamps.
  - Build a **uniform 50 Hz grid** over the time range both sensors cover, and linearly interpolate all axes onto it (`np.interp`).
  - When the source rate is above 50 Hz, apply an anti-alias Butterworth low-pass filter (~20 Hz) first.
  - Why 50 Hz: human motion energy lies mostly below ~20 Hz, and the Nyquist frequency is 25 Hz. Every recording can reach this rate, and window sizes in samples become identical.
- [ ] **Cleaning:** trim 1 s from each end of each clip; assert the cleaned length is 5–10 s; flag bad clips and re-record them.
- [ ] **Labelling:** add the `activity` column. Sessions get per-row labels expanded from their segment tables. Refine boundaries using the sync taps and the |a| plot.
- [ ] Save to `data/processed/train/...` using the A3 schema.
- [ ] **Figures:**
  - **Fig. 1 (report):** 4 panels, one clip per activity, |a| and |ω| against time.
  - Notebook only: all 6 axes per activity; Δt histogram before and after resampling; seconds-per-activity bar chart with a 90 s line.
- [ ] Sampling-rate summary table (configured vs measured vs harmonised) for the report.

**Commits**
1. `feat(notebook): add sampling quality checks`
2. `feat(notebook): add 50 Hz resampling with anti-alias filter`
3. `feat(notebook): add clip trimming and session labelling`
4. `data: add processed training clips and sessions`
5. `feat(notebook): add raw signal and sampling visualisations`

**⏸ Checkpoint 3**
- [ ] 52 processed clips, each 5–10 s, with no NaNs and Δt = 0.02 s throughout.
- [ ] Every activity totals ≥90 s (bar chart).
- [ ] Session labels line up with the visible signal changes.
- [ ] Fig. 1 is readable at report size and saved to `figures/`.
- [ ] The notebook runs top-to-bottom up to this point.

---

## Phase 4 — Unseen Test Data Collection
**Goal:** 2 truly unseen test sessions. **They must be recorded on a different day from the training data (Tue 27 Oct) and in a different place.**

**Why this design:**
- It matches the rubric's "2 test files".
- It exercises the transition matrix.
- It produces the decoded-sequence plot the brief asks for.
- It gives ~48 test windows per activity.
- A new day and place test generalisation honestly.

**Tasks**
- [ ] 2 sessions, **~100 s each, all 4 activities at ~25 s each**. Use activity orders that differ from the training sessions and from each other.
- [ ] Same protocol as the training sessions: sync taps, timer schedule, webcam footage, segment table.
- [ ] If the pilot showed the rate can be changed, record **test session 2 at a different configured rate** (e.g. half the training rate) to demonstrate harmonisation.
- [ ] Process both with the **unchanged** Phase 3 pipeline. Save them to `data/processed/test/` and update `metadata.csv`.

**Commits**
1. `data: add raw test sessions and segment labels`
2. `data: add processed test sessions`
3. `data: update metadata with test recordings`

**⏸ Checkpoint 4**
- [ ] Test recordings are dated after all training recordings, and the location type differs.
- [ ] Each test session contains all 4 activities; labels line up with the signal.
- [ ] The test files have not been looked at for tuning; only the label alignment was checked.

---

## Phase 5 — Windowing, Feature Extraction & Normalisation
**Goal:** a Z-scored feature matrix for training and test.

**Windowing:** **2.0 s = 100 samples at 50 Hz, hop 1.0 s.** The window is defined in seconds and converted with `fs × seconds`.
- Walking runs at ~1.6–2 steps/s and jumping at ~1.5–2.5 jumps/s, so a window covers ≥3 cycles.
- FFT resolution 1/T = 0.5 Hz.
- Expect ~175 training windows per activity.
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
| 13 | Band-energy ratio (0.5–3 / 3–15 Hz) | Freq | Gait energy is low-band; impacts and noise are high-band |
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
- [ ] Per-class box plots and a feature-correlation heatmap; drop any feature with |r| > 0.95 against another (keep D ≤ ~14).

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

**Verification** (notebook Section 5 "Model verification", all as assertions). Why: Implementation is worth 15 points; from-scratch HMM bugs are usually silent; and exact ground truth is stronger evidence than agreeing with a library.
1. **Brute-force likelihood:** for N = 2, T = 6 (64 paths), summing over every path must equal `exp(score)`.
2. **Brute-force Viterbi:** the argmax over all 64 paths must equal the Viterbi path.
3. **Forward–backward consistency:** P(X) from α must equal P(X) from β; each γ_t and ξ_t must sum to 1.
4. **EM monotonicity:** the LL must never decrease.
5. **Synthetic recovery:** sample from a known 3-state HMM, fit from a random start, match states, and check A and μ are within tolerance.
6. **Invariants:** rows of A and π sum to 1, and every variance is ≥ the floor, after every M-step.

**Tasks** (notebook Section 4 "HMM implementation" + Section 5)
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
1. Supervised initialisation from the labelled training windows (clips give self-transitions; sessions give cross-transitions).
2. Baum–Welch refinement over all training sequences.
3. State-to-label mapping check (Hungarian on the training confusion matrix).
4. Compare: **(a)** supervised only · **(b)** supervised + Baum–Welch · **(c)** per-window Gaussian classifier with no transitions (baseline). Baum–Welch maximises likelihood, not accuracy, so report whichever result happens and explain it.

**Evaluation**
- Decode both test sessions with Viterbi and pool the windows into a 4×4 confusion matrix (per-session numbers go in the notebook).
- Sensitivity = TP/(TP+FN); Specificity = TN/(TN+FP); per-row accuracy = (TP+TN)/N. Implement these in numpy.

| State (Activity) | Number of Samples (windows) | Sensitivity | Specificity | Overall Accuracy |
|---|---|---|---|---|
| Still | | | | |
| Standing | | | | |
| Walking | | | | |
| Jumping | | | | |
| **Overall** | total | macro avg | macro avg | correct / total |

  The footnote defines each formula and notes that one-vs-rest accuracy is inflated by true negatives, which is why the Overall row and the confusion matrix are the headline results.
- Generalisation: compare training accuracy with test accuracy; count errors within ±1 window of a transition vs errors inside a steady activity; discuss the effect of the new day, place, and (if done) rate.

**Report figures** (multi-panel, exported at 150–200 dpi to `figures/`)
- **Fig. 2:** transition-matrix heatmap (annotated) + emission-means heatmap (states × features).
- **Fig. 3:** test confusion matrix + Baum–Welch LL convergence curve.
- **Fig. 4:** decoded timelines (true vs predicted above |a|) for both test sessions.
- Notebook only: A before vs after Baum–Welch; Gaussian curves per state for 2–3 key features.

**Tasks** (notebook Section 6 "Training" + Section 7 "Evaluation")
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
- **Noise and sampling:** Android timing jitter, gyroscope drift, interpolation, aliasing, why 50 Hz is enough.
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
- [ ] ≥50 labelled clips (52 recorded; 5–10 s, timestamp column) + 2 training sessions; ≥90 s per activity
- [ ] 2 unseen continuous test sessions (new day and place) with segment labels
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
| Trivial transition matrix | Training sessions + Laplace smoothing | 2, 6 |
| Mislabelled session boundaries | Sync taps + timer + webcam; majority-label windows | 2, 4 |
| Underflow / singular variance | Scaled forward–backward; variance floor | 6 |
| Baum–Welch label switching | Supervised initialisation + Hungarian check | 7 |
| Leakage | Separate test day and place; Z-score fitted on train only; no tuning on test | 4, 5, 7 |
| Video links inaccessible | Incognito test; "anyone with link" sharing | 2, 8 |
| Privacy (public repo) | No GPS; videos not committed; `metadata.csv` says "indoor/outdoor", not addresses | 1, 2 |
| Over the page limit | 4-figure budget; extra plots stay in the notebook | 8 |
| Notebook fails for the grader | Relative paths; fresh-clone test; pinned requirements | 9 |
