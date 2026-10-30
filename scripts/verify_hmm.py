"""Independent exact and synthetic HMM verification; never load activity data."""

from itertools import product
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


if __name__ == '__main__':
    print(json.dumps(exact_verification(), indent=2))
