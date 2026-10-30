# Phase 5 — features and normalisation

Implemented in `scripts/features.py` and Section 4 of the executed notebook.
The fixed recording-level split is unchanged; no HMM fitting or test scoring has
been performed.

## Windowing

Use 2.0-second windows and a 1.0-second hop: 50/25 samples at 25 Hz. The two-second
support gives 0.5 Hz FFT-bin spacing and roughly 3–5 cycles for 1.5–2.5 Hz motion.
The last of 50 samples lies at 1.96 s; the window support is the half-open interval
[start,start+2 s). Sample stop and timestamp stop in the exported tables are
exclusive. Incomplete tails are discarded without padding.

Every clip or recovered session block is windowed independently, preserving its
original times and source ID. Majority labels use the fixed activity order to
resolve ties, with label purity recorded. All current labelled clips are pure
single-activity recordings. Unlabelled blocks remain unlabelled.

| Split | Independent sequences | Windows |
|---|---:|---:|
| Training clips | 44 | 338 |
| Held-out test clips | 8 | 65 |
| Optional unlabelled training blocks | 4 | 137 |

Overlapping windows are correlated observations, not independent recordings.

## Candidate features and definitions

Fourteen candidates are computed. Each justification is recorded in
`FEATURE_SPECS` and displayed in full in the notebook. Time-domain candidates are
three acceleration-axis means, population std of acceleration magnitude,
population variance of gyro magnitude, centred acceleration-magnitude RMS,
acceleration and gyro SMA, acceleration-magnitude range, and y/z acceleration
correlation. SMA is the mean sum of absolute axes. Correlation is zero when a
constant axis makes it undefined. Gravity remains present in acceleration, so
axis means and acceleration SMA are protocol-specific orientation cues as well
as activity descriptors.

FFT candidates are dominant positive frequency, spectral energy, low/high
band-energy ratio and normalised spectral entropy. Demean acceleration magnitude,
apply a periodic Hann taper, and compute `rfft`. Double interior positive power
bins, but not DC or the even-length Nyquist bin; divide by N times the squared
window sum. Here "squared window sum" means `sum(Hann**2)`, not `sum(Hann)**2`.
Summed power then equals the window-energy-corrected mean square. Peak and entropy
exclude DC; energy includes all bins. Entropy is divided by the log of the number
of positive-frequency bins. Bands are [0.5,3] Hz and (3,10] Hz, with no shared bin.

A total positive-frequency power of at most 1e-12 yields zero peak, ratio and
entropy; the high-band denominator is floored at 1e-12. This is a fixed numerical
convention, not a threshold selected from test data. A peak in low-power noise
must not be interpreted as a step rate. No clipped observations or fitted
spectral thresholds are used.

## Training-only screening and the final 10 features

The screen uses only the 338 labelled training windows. It removes constant
features, then scans the declared candidate order, dropping a later feature if
its absolute Pearson correlation with an already retained feature exceeds 0.95.
Earlier interpretable features take precedence. Every decision is saved in
`data/features/selection.csv`.

| Retained feature | Domain | Purpose |
|---|---|---|
| mean_acc_x | Time | Phone orientation |
| mean_acc_y | Time | Flat versus upright orientation cue |
| std_acc_mag | Time | Motion intensity |
| var_gyro_mag | Time | Variation in rotational movement |
| sma_acc | Time | Average absolute acceleration including gravity |
| sma_gyro | Time | Overall rotational movement |
| corr_acc_yz | Time | Coordination of two acceleration axes |
| dominant_freq_acc | FFT | Periodicity when combined with intensity |
| band_energy_ratio_acc | FFT | Low-frequency versus higher-frequency power |
| spectral_entropy_acc | FFT | Concentration versus spread of spectral power |

Removed: mean_acc_z (r≈−0.983 with mean_acc_y), centred RMS (r=1 with std_acc_mag),
acceleration range (r≈0.997 with std_acc_mag), and spectral energy (r≈0.988 with
std_acc_mag). Centred RMS and population standard deviation are mathematically
identical. The retained set has seven time-domain and three FFT-derived features;
all retained pairwise absolute training correlations are at most 0.95.

## Scaling and leakage controls

Fit per-feature population mean/std on the 338 labelled training windows only.
Transform test and optional unlabelled windows with those same vectors, without
refitting or clipping. Z-scoring improves numerical conditioning and makes the
later Gaussian variance floor comparable across mixed feature units. A Gaussian
with freely rescaled parameters is not inherently dominated by large units; this
is a numerical/regularisation choice, not a change to that mathematical property.

The transform stores fitted recording IDs, ordered features, the fixed split
hash, source/output hashes and code provenance. The test and unlabelled frames
are extracted only after the selection and scaler parameters are fixed. No test
feature-distribution plot, test-based feature choice or test performance metric
is produced. If later using training-only cross-validation, refit the selector
and scaler within each training fold; do not prefit them on the validation fold.

## Training observations

Median std(|a|) is approximately 0.0035, 0.0903, 0.7444 and 9.9793 m/s² for still,
standing, walking and jumping respectively. Mean a_y separates still (median
≈0.53 m/s²) from standing (≈9.61 m/s²) in this fixed-grip collection. Walking and
jumping have median dominant frequencies of 1.5 and 2 Hz. The 3.5 Hz median for
still/standing is low-energy fluctuation, not evidence of walking.

These are training observations, not accuracy claims. Orientation dependence,
the session grip difference, limited independent test clips, prior quality
inspection of all clips and absence of continuous test transitions remain the
previously documented limitations.

## Outputs and use

```bash
.venv/bin/python scripts/features.py
.venv/bin/python -m unittest discover -s tests -v
```

`data/features/{train,test,unlabelled}/` contains one normalised CSV per independent
sequence. Each CSV includes window provenance and the selected ten features;
labelled files also include activity and label purity. `transform.json` describes
feature order and statistics, `sequence_counts.csv` records counts, and
`training_candidates.csv` plus `training_correlations.csv` retain the training
screening evidence. `status.json` is marked ready only after generation succeeds.

Use `load_feature_sequences(ROOT, "train")` (or `"test"`, `"unlabelled"`) from
`scripts.features` for model-ready dictionaries: `name`, `X`, `y`, `windows`.
`X` is a (windows,10) matrix containing no timestamps, indices or labels. `y` is
encoded still=0, standing=1, walking=2, jumping=3, or None for unlabelled blocks.
The loader rejects stale code/split/input provenance and modified feature files.
Keep each returned sequence separate for later HMM training/decoding.

Training-only plots are exported as `figures/features_training_boxplots.png`
and `figures/features_training_correlation.png` and retained in the notebook.
They are notebook/supporting figures, not additions to the report's four-figure
budget. Analytical checks cover sinusoidal frequency/power, Parseval scaling,
Nyquist weighting, disjoint frequency bands, zero-energy inputs, window bounds,
majority ties and training-only transformations.
