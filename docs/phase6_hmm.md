# Phase 6 — HMM implementation and mathematical verification

Completed 30 October 2026. `scripts/hmm.py` implements a diagonal Gaussian HMM
from scratch using NumPy. The notebook displays its source in Section 5 and
demonstrates the verification assertions in Section 6. No activity features or
labels enter an HMM fit in this phase; real training and evaluation are Phase 7.

## Parameters and interface

For N states and D features, `pi` has shape (N,), `A` has shape (N,N), and
`means` and `variances` have shape (N,D). Rows of A and pi sum to one. All
variances must be at least `var_floor=1e-3`. The planned activity model uses N=4
and D=10; the verification examples deliberately use smaller synthetic models.

```python
from scripts.hmm import GaussianHMM

model = GaussianHMM(n_states=4, n_features=10)
# In Phase 7, supply lists of independently recorded TRAINING sequences only:
# model.fit_supervised(X_train, y_train)
# model.fit_baum_welch(X_train)
# path = model.predict_viterbi(X_one_recording)
# posterior = model.predict_proba(X_one_recording)
# total_log_density = model.score(X_sequences)
```

`set_parameters(pi, A, means, variances)` allows explicit initialisation and
validates/copies all parameters. Empty sequences, nonfinite data, incorrect
dimensions, invalid probabilities and variances below the floor fail explicitly.
`score` and both fitting methods require a list of sequences, even for one file.
`predict_proba` and `predict_viterbi` accept one (T,D) array.

## Numerical conventions

The emission log density includes the full Gaussian normalising constant:

`log B[t,j] = -0.5 * (D*log(2*pi) + sum(log(var[j])) + sum((X[t]-mean[j])²/var[j]))`.

Forward/backward scaling is performed in log space. `_forward(log_B)` returns
normalised log-alpha messages (T,N), log scaling constants (T,), and their sum,
the total log likelihood. Each constant is a conditional observation log density.
`_backward(log_B, log_scales)` subtracts the next forward log scale at every
recursion and ends at log-beta=0. Log-sum-exp uses a maximum shift; exact zero
probabilities remain `-inf`. Gaussian densities can exceed one, so positive log
likelihoods are valid. Extremely large observations whose squared deviations
overflow floating point raise an error rather than producing unusable results.

Gamma is normalised from log-alpha + log-beta. Xi uses the current alpha, A,
the next emission and next beta, and sums to one at each within-sequence
transition. Singleton sequences have a gamma but no xi entries.

Viterbi maximises joint path log probability and backtracks through predecessor
indices. It does not select each window's marginal posterior independently.
Exact ties choose the lowest predecessor/final state index deterministically.

## Initialisation and EM

Supervised initialisation pools labelled observations only for emission means
and population variances. Every state must have labelled observations. It
counts one initial state per sequence and transitions only within each sequence.
Default Laplace pseudocounts add one to each initial/transition count. Explicit
positive arrays can instead encode a documented persistence prior. The arrays
and actual counts are retained in `supervised_counts_`. A cross-activity prior
is an assumption, not evidence from the approximate session annotations.

Baum–Welch independently evaluates each sequence and sums its expected counts.
Initial probabilities average its first-state posteriors; transition rows use
within-sequence xi counts. Weighted emission means and variances use gamma.
Variances are computed around the new means to avoid subtracting two large
second moments, then floored. States with zero occupancy and rows with no
expected outgoing transitions retain their previous emission/transition values.
This also handles collections of singleton sequences without inventing edges.

EM uses maximum likelihood with a variance constraint. Initialisation's Laplace
counts are **not** reapplied in EM: doing that would change the objective being
checked for monotonicity. Exact zero transitions remain structural zeros.

`history_` starts with the initial total log likelihood and includes each
accepted M-step. Stop when the absolute change is below `tol=1e-4`; the default
500 iterations is a safety cap. `converged_`, `stop_reason_` and `n_iter_` make
the distinction explicit. Parameter invariants are checked after every M-step
and recorded in `invariant_history_`. A decrease beyond
`1e-9 + 100*machine_epsilon*abs(previous_LL)` raises and restores the previous
parameters. A smaller roundoff decrease is rolled back and reported as
`roundoff_plateau`, with zero accepted improvement. The recovery example stops
normally on `likelihood_tolerance`, with no rollback.

## Verification evidence

Run `python scripts/verify_hmm.py` to regenerate `data/verification/hmm.json`.
The file includes source hashes, NumPy version, seeds, model parameters,
likelihood history and every M-step's invariant diagnostics.

1. **Exact likelihood:** enumerate all 64 paths for N=2,T=6 using an independent
   ordinary Gaussian PDF. Forward likelihood differs from the sum by
   approximately 2.71e-20.
2. **Exact Viterbi:** the best enumerated path and decoder both return
   `[0, 0, 1, 1, 1, 0]`.
3. **Forward/backward:** an independent unscaled backward recurrence agrees with
   the path sum. Gamma and xi match enumeration within 2.23e-16, normalise to
   one and have consistent adjacent marginals. An entire M-step also matches
   independently enumerated sufficient statistics.
4. **Monotonic EM:** every accepted synthetic update has nonnegative improvement.
   A deliberately broken M-step in the regression tests proves that a real
   likelihood decrease is rejected and parameters are restored.
5. **Synthetic recovery:** twelve independently sampled 250-step sequences from
   a known three-state, two-feature HMM; data seed 2026, initialisation seed 7.
   One random start uses Dirichlet probabilities, means uniform within observed
   bounds, and pooled variances. No hidden labels are supplied to fitting.
   After fitting, all six state permutations are considered to align means
   with truth; A is permuted on both axes. Maximum errors are 0.01421 for A
   (limit 0.06) and 0.03212 for means (limit 0.15). EM converges in 7 iterations,
   with final improvement 3.82e-11, below 1e-4 and well before the 500-step cap.
6. **Invariants:** pi and all A rows sum to one, all parameters remain finite,
   and variances meet the floor after every update.

The complete regression suite passes all 54 tests. Additional checks cover singleton sequences, zero occupancy, exact
zero probabilities, long low-density sequences, positive Gaussian log densities,
variance flooring, supervised counts and the difference between convergence
and hitting an iteration cap. Run `python -m unittest discover -s tests -v`.

## Interpretation limits

This is numerical verification on deliberately identifiable synthetic states,
not activity-recognition accuracy or evidence of global EM convergence. One
successful random start does not establish that every start succeeds. A and
means are the stated recovery targets; the initial distribution has only twelve
independent starts and is not given a parameter-recovery claim. State matching
uses known synthetic means only for verification; future activity state mapping
must use training labels. The real-data test split remains reserved for Phase 7.
