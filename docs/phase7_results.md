# Phase 7 — activity training and held-out evaluation

> **Update, 31 October 2026.** A final review found that free Baum–Welch made still
> and jumping absorbing (A = 1.000), because every training clip contains one
> activity. The primary model now keeps the supervised transition prior fixed
> during EM (pi and emissions are still re-estimated): 7 updates, training LL
> 654.82 → 664.10. The free-A model described below is kept as the comparison
> `baum_welch_free_transitions`. Held-out predictions are identical (63/65), and
> Figure 4 now decodes the unlabelled session blocks. Numbers below that describe
> EM (8 updates, LL 674.29, absorbing rows) refer to the free-A comparison.

Completed 30 October 2026. Notebook Sections 7–8 reproduce training, the fixed
three-model comparison, the rubric metrics table, per-clip results, report
Figures 2–4, and the notebook-only parameter plots.

## Protocol fixed before test evaluation

The primary model is **supervised initialisation followed by Baum–Welch**, chosen
before test metrics were inspected. Comparators are the supervised-only HMM and
an independent per-window Gaussian MAP classifier. All three use the same ten
features and training-fitted z-score transform. No features, variance floors,
transition priors, stopping settings, state maps or primary-model choices were
changed after examining test predictions.

`scripts/train_activity.py` fits only the 44 labelled training clips (338
windows). Every recording remains an independent sequence. Defaults follow the
plan: four states, diagonal Gaussian emissions, variance floor 1e-3, absolute
likelihood tolerance 1e-4, and 500 iterations as a safety cap. Supervised initial
and transition pseudocounts are one per entry. Actual transition counts are
diagonal `[88,77,69,60]`; cross-activity transition probabilities at initialisation therefore come from
the explicit smoothing assumption, not verified continuous-session transitions.

The four optional unlabelled blocks are excluded a priori because of the
documented session-grip change and uncertainty about their activity boundaries.
They remain available for a future separately validated experiment. No optional
variant was compared on test data. No hyperparameter search is performed.

For each HMM, Hungarian matching maximises agreement between **training**
Viterbi state IDs and known training activity labels. Both maps are identity:
still=0, standing=1, walking=2, jumping=3. These maps are frozen for test decoding.
The Gaussian baseline uses supervised emission log densities plus the same
supervised initial log probabilities independently at every window. Its labelled
emissions retain their supervised identity; it uses neither A nor neighbours.

The training protocol, model parameters, raw training counts, feature order,
source/input hashes, mappings and EM diagnostics were saved to
`data/models/activity_models.json` and committed before the first test evaluation.
Training likelihood rises from **654.817634** to **674.289539** in **8 updates**;
the final improvement is **3.58964e-5**, below 1e-4. All updates satisfy parameter
invariants and nondecreasing likelihood. Positive continuous log density is
valid; it is not a probability or a classification accuracy.

## Fixed comparison

| Model | Training correct / windows | Training accuracy | Test correct / windows | Test accuracy |
|---|---:|---:|---:|---:|
| Supervised HMM | 337 / 338 | 99.70% | 64 / 65 | 98.46% |
| Baum–Welch HMM (preselected primary) | 336 / 338 | 99.41% | 63 / 65 | 96.92% |
| Independent Gaussian baseline | 336 / 338 | 99.41% | 64 / 65 | 98.46% |

Training scores are resubstitution scores, not independent validation estimates.
EM improves likelihood but slightly worsens classification: it makes one extra
test error. The baseline ties the supervised HMM here, so these clips do not
demonstrate a test-accuracy benefit from temporal modelling. The primary model
is retained as specified before evaluation; the better comparator results are
reported openly, without a new selection or tuning cycle.

## Primary model metrics

Rows below use true activities; confusion columns use predicted activities in
the same order: still, standing, walking, jumping.

```
[[18,  0,  0,  0],
 [ 0, 16,  2,  0],
 [ 0,  0, 18,  0],
 [ 0,  0,  0, 11]]
```

| State (Activity) | Number of Samples (windows) | Sensitivity | Specificity | Overall Accuracy* |
|---|---:|---:|---:|---:|
| Still | 18 | 100.00% | 100.00% | 100.00% |
| Standing | 18 | 88.89% | 100.00% | 96.92% |
| Walking | 18 | 100.00% | 95.74% | 96.92% |
| Jumping | 11 | 100.00% | 100.00% | 100.00% |
| **Overall** | **65** | **97.22%** | **98.94%** | **96.92%** |

Sensitivity = TP/(TP+FN); specificity = TN/(TN+FP).
*Per-activity accuracy = (TP+TN)/65, a one-vs-rest value inflated by true
negatives. The Overall row uses macro-average sensitivity/specificity and true
multiclass accuracy = trace(confusion)/65. These quantities are implemented in
NumPy and checked against hand-computed examples; undefined denominators are
reported as NaN rather than silently assigning a perfect or zero score.

## Per-clip findings

All mistakes occur in `standing_06`. The primary model gets 7/9 windows right;
the other seven clips are all correct. The supervised model and baseline each
get 8/9 on `standing_06`. With equal weight per clip, primary accuracy is 97.22%;
the headline window-pooled accuracy is 96.92% because clip lengths differ.

| Primary error window (zero-based) | Original-recording interval | Predicted activity |
|---|---|---|
| 3 | [5.445213, 7.445213) s | Walking |
| 8 | [10.445213, 12.445213) s | Walking |

Window 3 is also wrong for both comparators; window 8 is the additional EM
error. The true retained label remains standing. The errors do not establish
that the participant was actually walking, and labels are not edited to match
predictions. Window 3 coincides with a local fluctuation in the plotted signal;
the model also uses nine other features, so that visual association alone does
not establish a cause. The final error is at a sequence end, which has no later
observations to constrain its decoded state. No cross-activity test transition
is present, so transition-boundary error rates cannot be measured.

## Figures and interpretation

- **Figure 2 — `fig2_transition_emissions.png`:** primary A and Gaussian means in
  training-standardised coordinates. The mean heatmap is a parameter summary,
  not a categorical emission-probability table. Still separates by phone
  orientation; jumping has high motion-intensity features. After EM, still and
  jumping have effectively absorbing rows, while standing/walking retain small
  cross-probabilities. This reflects the single-activity training design and
  emission ambiguity; it is not evidence of real elderly activity dynamics.
  EM has no probability pseudocounts, so initial cross-state probabilities can
  collapse to zero. Values printed as 0.000 may also be tiny positive numbers.
- **Figure 3 — `fig3_confusion_convergence.png`:** the primary held-out confusion
  matrix beside training-only likelihood convergence. Likelihood increases do
  not imply better classification; the comparison table shows the difference.
- **Figure 4 — `fig4_test_timelines.png`:** one independent held-out clip per
  activity, chosen by the predeclared lexicographically-first-name rule:
  still_02, standing_06, walking_02, jumping_02. True/predicted labels are shown
  at the centres of overlapping two-second windows, above processed acceleration
  magnitude. Time is relative to each retained clip's start. Steps connect
  consecutive decisions within that file only and do not invent activity
  transitions. Magnitude axes use separate ranges; note the units and scales.

All report figures are exported at 180 dpi. The notebook additionally shows A
before/after EM, Gaussian marginals for mean acc_y, std |a| and dominant
frequency, and separate timelines for the other four test clips.

## Scope and remaining limits

There are only eight independent test clips, two per class, from the same person,
phone and collection conditions as training. Their 65 overlapping windows are
not 65 independent trials; no binomial confidence interval based on that
assumption is claimed. All clips were quality-inspected/cropped before the split,
as previously disclosed. After splitting, fitting and state mapping use training
only; test features and labels are used only for final prediction and metrics.

The high scores support narrow same-collection single-activity recognition.
They do not establish generalisation to a different day, environment, person,
device, phone placement, or native sampling rate. The flat-phone/upright-grip
orientation cue is protocol-specific. Jumping is an impact activity, not a
validated fall class; this is not validated elderly monitoring. The held-out
clips cannot test realistic cross-activity dynamics, duration models, or an
impact-then-inactivity alert. Additional independent continuous recordings are
needed for those claims.

## Reproduction and artifacts

```bash
python scripts/train_activity.py
python scripts/evaluate_activity.py
python scripts/activity_plots.py
python -m unittest discover -s tests -v
```

`data/evaluation/` contains model comparisons, per-class/per-clip metrics,
confusion matrices, each window's true/predicted state with original timing,
and a summary binding results to the frozen model and test feature hashes.
Re-running replays this protocol; it does not select a new model. All 63 regression tests pass. The separation
tests reject test/unlabelled feature loading during training and forbid any fit
or state remapping during evaluation. Changing test labels in an isolated test
fixture changes metrics but leaves predictions and model bytes unchanged.

Phase 7 is complete. Phase 8 assembles the report from these artifacts and
must preserve their comparison and limitations rather than imply EM improved
accuracy or that clinical/transition performance has been established.
