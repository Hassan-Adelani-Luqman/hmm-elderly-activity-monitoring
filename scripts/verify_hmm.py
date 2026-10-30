"""Independent exact and synthetic HMM verification; never load activity data."""

from itertools import product, permutations
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

if __package__:
    from .hmm import GaussianHMM
else:
    from hmm import GaussianHMM


def exact_verification():
    """Enumerate 64 paths of a two-state, six-step toy; assert LL/path/gamma/xi."""
    model = GaussianHMM(2, 1).set_parameters(
        [.6, .4], [[.75, .25], [.3, .7]], [[-1.], [1.5]], [[.5], [.8]])
    X = np.array([[-1.2], [-.4], [.7], [1.8], [1.1], [-.6]])
    # Independently evaluate the ordinary Gaussian PDF, not _log_emission().
    B = np.empty((len(X), 2))
    for t, observation in enumerate(X):
        for state in range(2):
            B[t, state] = np.prod(np.exp(-.5*(observation-model.means[state])**2/model.variances[state])
                                   / np.sqrt(2*np.pi*model.variances[state]))
    paths = list(product(range(2), repeat=len(X)))
    probabilities = []
    for path in paths:
        weight = model.pi[path[0]] * B[0, path[0]]
        for t in range(1, len(X)):
            weight *= model.A[path[t-1], path[t]] * B[t, path[t]]
        probabilities.append(weight)
    probabilities = np.array(probabilities)
    total = probabilities.sum()
    expected_gamma = np.zeros_like(B)
    expected_xi = np.zeros((len(X)-1, 2, 2))
    for path, probability in zip(paths, probabilities/total):
        for t, state in enumerate(path):
            expected_gamma[t, state] += probability
        for t in range(len(X)-1):
            expected_xi[t, path[t], path[t+1]] += probability
    gamma, xi, ll = model._posteriors(X)
    np.testing.assert_allclose(np.exp(model.score([X])), total, rtol=1e-12, atol=0)
    expected_path = np.array(paths[int(np.argmax(probabilities))])
    np.testing.assert_array_equal(model.predict_viterbi(X), expected_path)
    np.testing.assert_allclose(gamma, expected_gamma, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(xi, expected_xi, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(gamma.sum(axis=1), 1, atol=1e-13)
    np.testing.assert_allclose(xi.sum(axis=(1, 2)), 1, atol=1e-13)
    np.testing.assert_allclose(xi.sum(axis=2), gamma[:-1], atol=1e-13)
    np.testing.assert_allclose(xi.sum(axis=1), gamma[1:], atol=1e-13)
    # An independent unscaled backward recurrence gives another P(X).
    beta = np.ones_like(B)
    for t in range(len(X)-2, -1, -1):
        for i in range(2):
            beta[t, i] = sum(model.A[i, j]*B[t+1, j]*beta[t+1, j] for j in range(2))
    backward_likelihood = np.sum(model.pi * B[0] * beta[0])
    np.testing.assert_allclose(backward_likelihood, total, rtol=1e-12, atol=0)
    log_B = model._log_emission(X)
    alpha, scales, _ = model._forward(log_B)
    scaled_beta = model._backward(log_B, scales)
    np.testing.assert_allclose(np.exp(alpha + scaled_beta).sum(axis=1), 1, atol=1e-12)
    # Independent enumerated sufficient statistics also verify a complete EM step.
    counts = expected_xi.sum(axis=0)
    weights = expected_gamma.sum(axis=0)
    means = expected_gamma.T @ X / weights[:, None]
    variances = np.array([np.sum(expected_gamma[:, j, None]*(X-means[j])**2, axis=0)/weights[j] for j in range(2)])
    expected_A = counts/counts.sum(axis=1, keepdims=True)
    step = GaussianHMM(2, 1, max_iter=1).set_parameters(model.pi, model.A, model.means, model.variances)
    step.fit_baum_welch([X])
    np.testing.assert_allclose(step.pi, expected_gamma[0], atol=1e-12)
    np.testing.assert_allclose(step.A, expected_A, atol=1e-12)
    np.testing.assert_allclose(step.means, means, atol=1e-12)
    np.testing.assert_allclose(step.variances, np.maximum(variances, step.var_floor), atol=1e-12)
    return {"paths_enumerated": len(paths), "brute_force_likelihood": float(total),
            "forward_log_likelihood": ll, "backward_likelihood": float(backward_likelihood),
            "likelihood_absolute_error": float(abs(np.exp(ll)-total)),
            "viterbi_path": expected_path.tolist(),
            "maximum_gamma_error": float(np.max(abs(gamma-expected_gamma))),
            "maximum_xi_error": float(np.max(abs(xi-expected_xi))),
            "exact_em_step_passed": True}


def sample_sequences(pi, A, means, variances, lengths, seed):
    """Sample independent Gaussian HMM X_s(T_s,D) arrays and hidden z_s(T_s,)."""
    rng = np.random.default_rng(seed)
    observations, states = [], []
    for length in lengths:
        z = np.empty(length, dtype=np.int64)
        X = np.empty((length, means.shape[1]))
        z[0] = rng.choice(len(pi), p=pi)
        for t in range(length):
            if t:
                z[t] = rng.choice(len(pi), p=A[z[t-1]])
            X[t] = rng.normal(means[z[t]], np.sqrt(variances[z[t]]))
        observations.append(X)
        states.append(z)
    return observations, states


def synthetic_verification():
    """Recover a known 3-state/2-feature HMM from one seeded random initialisation.

    Seeds, sample lengths and recovery tolerances are fixed. Hidden labels are
    not supplied to the fit. All 3! state permutations are considered only after
    training, to compare estimated parameters with their known synthetic truth.
    """
    pi = np.array([.5, .3, .2])
    A = np.array([[.91, .06, .03], [.04, .90, .06], [.05, .05, .90]])
    means = np.array([[-4., -2.], [0., 3.], [4., -1.]])
    variances = np.array([[.35, .5], [.45, .3], [.4, .6]])
    data_seed, initial_seed = 2026, 7
    lengths = [250]*12
    X_list, _ = sample_sequences(pi, A, means, variances, lengths, data_seed)
    pooled = np.concatenate(X_list)  # Initial emission scale only; never an HMM sequence.
    rng = np.random.default_rng(initial_seed)
    model = GaussianHMM(3, 2).set_parameters(
        rng.dirichlet(np.ones(3)), rng.dirichlet(np.ones(3), size=3),
        rng.uniform(pooled.min(axis=0), pooled.max(axis=0), size=(3, 2)),
        np.tile(pooled.var(axis=0), (3, 1)))
    initial_means = model.means.copy()
    model.fit_baum_welch(X_list)
    permutation = np.array(min(permutations(range(3)), key=lambda p: np.sum((model.means[list(p)]-means)**2)))
    matched_A = model.A[np.ix_(permutation, permutation)]
    matched_means = model.means[permutation]
    A_error = float(np.max(abs(matched_A-A)))
    means_error = float(np.max(abs(matched_means-means)))
    increments = np.diff(model.history_)
    assert model.converged_ and model.stop_reason_ == 'likelihood_tolerance'
    assert model.n_iter_ < model.max_iter
    assert (increments >= 0).all()
    assert abs(increments[-1]) < model.tol
    assert A_error < .06, f'Transition recovery error {A_error}'
    assert means_error < .15, f'Mean recovery error {means_error}'
    assert len(model.invariant_history_) == model.n_iter_
    for row in model.invariant_history_:
        assert row['pi_sum_error'] <= 1e-12
        assert row['A_row_sum_error'] <= 1e-12
        assert row['minimum_variance'] >= model.var_floor
    return {'data_seed': data_seed, 'initial_seed': initial_seed, 'sequence_lengths': lengths,
            'random_starts': 1, 'truth': {'pi': pi.tolist(), 'A': A.tolist(),
                                         'means': means.tolist(), 'variances': variances.tolist()},
            'initial_means': initial_means.tolist(), 'matched_state_indices': permutation.tolist(),
            'matched_A': matched_A.tolist(), 'matched_means': matched_means.tolist(),
            'matched_variances': model.variances[permutation].tolist(),
            'maximum_A_error': A_error, 'A_error_tolerance': .06,
            'maximum_mean_error': means_error, 'mean_error_tolerance': .15,
            'converged': model.converged_, 'stop_reason': model.stop_reason_,
            'iterations': model.n_iter_, 'iteration_cap': model.max_iter, 'tol': model.tol,
            'history': model.history_, 'invariant_history': model.invariant_history_,
            'final_improvement': float(increments[-1]), 'minimum_improvement': float(increments.min())}


def run_verification(output_path=None):
    """Run exact and synthetic assertions; optionally save JSON evidence at path."""
    result = {'passed': True, 'scope': 'Synthetic data only; no activity model fitted or scored',
              'numpy_version': np.__version__,
              'source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in ['hmm.py', 'verify_hmm.py']},
              'exact': exact_verification(), 'synthetic': synthetic_verification()}
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'data/verification/hmm.json')
    args = parser.parse_args()
    result = run_verification(args.output)
    synthetic = result['synthetic']
    print(json.dumps({'passed': result['passed'], 'iterations': synthetic['iterations'],
                      'final_improvement': synthetic['final_improvement'],
                      'maximum_A_error': synthetic['maximum_A_error'],
                      'maximum_mean_error': synthetic['maximum_mean_error'],
                      'evidence': str(args.output)}, indent=2))
