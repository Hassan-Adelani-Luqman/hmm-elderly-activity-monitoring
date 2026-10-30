"""From-scratch NumPy diagonal Gaussian HMM; sequences remain independent.

Forward/backward messages are scaled in log space: log_scales[t] is the log
predictive density at t, and its sum is the sequence log likelihood. No HMM
library or real activity data is used by this module's verification examples.
"""

import numpy as np


def logsumexp(values, axis=None):
    """Stable log(sum(exp(values))) along axis; all -inf slices stay -inf."""
    values = np.asarray(values, dtype=float)
    maximum = np.max(values, axis=axis, keepdims=True)
    safe = np.where(np.isfinite(maximum), maximum, 0.0)
    with np.errstate(divide="ignore"):
        result = safe + np.log(np.sum(np.exp(values - safe), axis=axis, keepdims=True))
    return result.squeeze() if axis is None else np.squeeze(result, axis=axis)


def log_probabilities(probabilities):
    """Map probability array to logs, preserving exact zeros as -inf."""
    with np.errstate(divide="ignore"):
        return np.log(probabilities)


class GaussianHMM:
    """N states, D features, parameters pi(N), A(N,N), means/variances(N,D)."""

    def __init__(self, n_states, n_features, var_floor=1e-3, tol=1e-4, max_iter=500):
        """Configure positive N,D, variance floor, absolute LL tolerance and cap."""
        for name, value in [("n_states", n_states), ("n_features", n_features), ("max_iter", max_iter)]:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(var_floor) or var_floor <= 0 or not np.isfinite(tol) or tol <= 0:
            raise ValueError("var_floor and tol must be finite and positive")
        self.n_states, self.n_features = n_states, n_features
        self.var_floor, self.tol, self.max_iter = float(var_floor), float(tol), max_iter
        self.pi = self.A = self.means = self.variances = None

    def set_parameters(self, pi, A, means, variances):
        """Copy and validate pi(N), A(N,N), means(N,D), variances(N,D); return self."""
        previous = self.pi, self.A, self.means, self.variances
        self.pi, self.A, self.means, self.variances = [
            np.array(p, dtype=float, copy=True) for p in (pi, A, means, variances)]
        try:
            self._validate_parameters()
        except ValueError:
            self.pi, self.A, self.means, self.variances = previous
            raise
        return self

    def _validate_parameters(self):
        """Check finite parameter shapes, probability sums and variance bounds."""
        for value, shape in [(self.pi, (self.n_states,)), (self.A, (self.n_states, self.n_states)),
                             (self.means, (self.n_states, self.n_features)),
                             (self.variances, (self.n_states, self.n_features))]:
            if value is None or value.shape != shape or not np.isfinite(value).all():
                raise ValueError("Initialise finite parameters with the required shapes")
        if (self.pi < 0).any() or (self.A < 0).any():
            raise ValueError("Probabilities must be nonnegative")
        if not np.isclose(self.pi.sum(), 1, atol=1e-12, rtol=0) or not np.allclose(self.A.sum(axis=1), 1, atol=1e-12, rtol=0):
            raise ValueError("pi and each row of A must sum to one")
        if (self.variances < self.var_floor).any():
            raise ValueError("Variances must meet var_floor")

    def _validate_X(self, X):
        """Return finite nonempty float observations of shape (T,D)."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features or len(X) == 0 or not np.isfinite(X).all():
            raise ValueError("X must be a nonempty finite (T,D) array")
        return X

    def _sequences(self, X_list):
        """Validate a nonempty list of separate (T_s,D) arrays; never concatenate."""
        if isinstance(X_list, np.ndarray):
            raise ValueError("Pass a list of sequences, including for a single recording")
        sequences = [self._validate_X(X) for X in X_list]
        if not sequences:
            raise ValueError("At least one sequence is required")
        return sequences

    def _log_emission(self, X):
        """Diagonal Gaussian log densities (T,N) for X(T,D), including constants."""
        self._validate_parameters()
        X = self._validate_X(X)
        with np.errstate(over="ignore"):
            result = -0.5 * (self.n_features * np.log(2 * np.pi)
                             + np.log(self.variances).sum(axis=1)
                             + np.sum((X[:, None, :] - self.means) ** 2 / self.variances, axis=2))
        if not np.isfinite(result).all():
            raise ValueError("Gaussian log density exceeds floating-point range")
        return result

    def _validate_log_B(self, log_B):
        """Validate emission log weights (T,N); permit impossible states (-inf)."""
        log_B = np.asarray(log_B, dtype=float)
        if (log_B.ndim != 2 or log_B.shape[1] != self.n_states or len(log_B) == 0
                or np.isnan(log_B).any() or np.isposinf(log_B).any()):
            raise ValueError("log_B must be (T,N) without NaN or +inf")
        return log_B

    def _forward(self, log_B):
        """Return normalised log_alpha(T,N), log_scales(T), scalar log likelihood."""
        self._validate_parameters()
        log_B = self._validate_log_B(log_B)
        log_A = log_probabilities(self.A)
        alpha = np.empty_like(log_B)
        scales = np.empty(len(log_B))
        for t in range(len(log_B)):
            predicted = log_probabilities(self.pi) if t == 0 else logsumexp(alpha[t-1, :, None] + log_A, axis=0)
            row = predicted + log_B[t]
            scales[t] = logsumexp(row)
            if not np.isfinite(scales[t]):
                raise ValueError("Observation sequence has no finite-probability path")
            alpha[t] = row - scales[t]
        return alpha, scales, float(scales.sum())

    def _backward(self, log_B, log_scales):
        """Return scaled log_beta(T,N), using forward log_scales(T); terminal zero."""
        self._validate_parameters()
        log_B = self._validate_log_B(log_B)
        log_scales = np.asarray(log_scales, dtype=float)
        if log_scales.shape != (len(log_B),) or not np.isfinite(log_scales).all():
            raise ValueError("Expected finite forward log_scales(T)")
        beta = np.zeros_like(log_B)
        log_A = log_probabilities(self.A)
        for t in range(len(log_B) - 2, -1, -1):
            beta[t] = logsumexp(log_A + log_B[t+1] + beta[t+1], axis=1) - log_scales[t+1]
        return beta

    def _posteriors(self, X):
        """Return gamma(T,N), xi(T-1,N,N), LL; no transitions across sequences."""
        log_B = self._log_emission(X)
        alpha, scales, likelihood = self._forward(log_B)
        beta = self._backward(log_B, scales)
        weights = alpha + beta
        gamma = np.exp(weights - logsumexp(weights, axis=1)[:, None])
        xi = np.empty((len(log_B)-1, self.n_states, self.n_states))
        for t in range(len(xi)):
            weights = alpha[t, :, None] + log_probabilities(self.A) + log_B[t+1] + beta[t+1]
            xi[t] = np.exp(weights - logsumexp(weights))
        return gamma, xi, likelihood

    def score(self, X_list):
        """Sum independent sequence log densities for a list of (T_s,D) arrays."""
        return float(sum(self._forward(self._log_emission(X))[2] for X in self._sequences(X_list)))

    def predict_proba(self, X):
        """Return smoothed state posterior gamma(T,N) for one X(T,D) sequence."""
        return self._posteriors(X)[0]

    def predict_viterbi(self, X):
        """Return the most likely joint state path (T,) for X(T,D), in log space.

        Backpointers select the whole path, not per-window posterior maxima.
        Exact ties choose the lowest predecessor/final state index.
        """
        log_B = self._log_emission(X)
        log_A = log_probabilities(self.A)
        delta = log_probabilities(self.pi) + log_B[0]
        back = np.zeros(log_B.shape, dtype=np.int64)
        for t in range(1, len(log_B)):
            candidates = delta[:, None] + log_A
            back[t] = np.argmax(candidates, axis=0)
            delta = candidates[back[t], np.arange(self.n_states)] + log_B[t]
        path = np.empty(len(log_B), dtype=np.int64)
        path[-1] = np.argmax(delta)
        for t in range(len(log_B)-1, 0, -1):
            path[t-1] = back[t, path[t]]
        return path
