"""Phase 7 training: fixed protocol, labelled training clips only, no test scoring."""

from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.optimize import linear_sum_assignment

if __package__:
    from .features import load_feature_sequences
    from .hmm import GaussianHMM
else:
    from features import load_feature_sequences
    from hmm import GaussianHMM


PROTOCOL = {
    "primary_model": "baum_welch",
    "models": ["supervised", "baum_welch", "baum_welch_free_transitions", "gaussian_baseline"],
    "n_states": 4, "var_floor": 1e-3, "tol": 1e-4, "max_iter": 500,
    "transition_pseudocount": 1.0, "initial_pseudocount": 1.0,
    "em_update_transitions": False,
    "transition_policy": "Every labelled training clip contains one activity, so it carries no evidence about "
                         "activity changes. Free EM drives every unobserved transition to exactly zero, making "
                         "still and jumping absorbing. EM therefore keeps the supervised Laplace transition "
                         "prior fixed and re-estimates pi and the Gaussian emissions only. Decided from the "
                         "fitted parameters, not from test metrics. The free-transition EM model is kept "
                         "as a comparison only",
    "training_split": "train", "include_unlabelled": False,
    "unlabelled_policy": "Excluded a priori: unverified boundaries and changed grip; no auxiliary variant selected using test results",
    "mapping_policy": "Hungarian matching on labelled-training Viterbi predictions only",
    "baseline_policy": "Supervised Gaussian log density plus supervised initial log probability independently at every window; original supervised label identity",
    "selection_policy": "Fixed planned parameters and primary Baum-Welch model; no hyperparameter search or model selection using test metrics",
    "timeline_policy": "Lexicographically first held-out recording per activity, chosen without inspecting predictions",
}


def file_hash(path):
    """Return SHA-256 of one artifact's exact bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_mapping(y_sequences, predicted_states, n_states):
    """Hungarian train-only mapping from states to labels; return map(N), counts(N,N)."""
    if len(y_sequences) != len(predicted_states) or not y_sequences:
        raise ValueError("Provide paired, nonempty training sequences")
    counts = np.zeros((n_states, n_states), dtype=np.int64)
    for y, states in zip(y_sequences, predicted_states):
        y, states = np.asarray(y), np.asarray(states)
        if (y.ndim != 1 or y.shape != states.shape or not len(y)
                or not np.issubdtype(y.dtype, np.integer) or not np.issubdtype(states.dtype, np.integer)
                or (y < 0).any() or (y >= n_states).any() or (states < 0).any() or (states >= n_states).any()):
            raise ValueError("Expected aligned integer labels/states in [0,N)")
        np.add.at(counts, (y, states), 1)
    if (counts.sum(axis=1) == 0).any():
        raise ValueError("Mapping needs labelled training support for every activity")
    labels, states = linear_sum_assignment(-counts)
    mapping = np.empty(n_states, dtype=np.int64)
    mapping[states] = labels
    return mapping, counts


def parameters(model):
    """Serialise the four Gaussian HMM parameter arrays to JSON-compatible lists."""
    return {name: getattr(model, name).tolist() for name in ['pi', 'A', 'means', 'variances']}


def model_from_parameters(record, protocol=PROTOCOL):
    """Restore a validated HMM from a model record; no fitting or label mapping."""
    values = record['parameters']
    return GaussianHMM(protocol['n_states'], len(values['means'][0]),
                       var_floor=protocol['var_floor'], tol=protocol['tol'],
                       max_iter=protocol['max_iter']).set_parameters(
                           values['pi'], values['A'], values['means'], values['variances'])


def train_models(root):
    """Fit supervised and EM models using only independent labelled train (T,D) arrays.

    Persist frozen parameters and mappings before evaluation can load test features.
    The primary model and optional-data policy are fixed in PROTOCOL, not selected
    by training resubstitution accuracy or final test performance.
    """
    root = Path(root)
    train = load_feature_sequences(root, 'train')
    config = json.loads((root/'data/features/transform.json').read_text())
    if len(train) != 44 or sum(len(s['X']) for s in train) != 338:
        raise ValueError("Expected the fixed 44-clip/338-window training collection")
    X, y = [s['X'] for s in train], [s['y'] for s in train]
    supervised = GaussianHMM(PROTOCOL['n_states'], len(config['feature_order']),
                             var_floor=PROTOCOL['var_floor'], tol=PROTOCOL['tol'], max_iter=PROTOCOL['max_iter'])
    supervised.fit_supervised(X, y, transition_prior=PROTOCOL['transition_pseudocount'],
                              initial_prior=PROTOCOL['initial_pseudocount'])
    refined = model_from_parameters({'parameters': parameters(supervised)})
    refined.fit_baum_welch(X, update_transitions=PROTOCOL['em_update_transitions'])
    free = model_from_parameters({'parameters': parameters(supervised)})
    free.fit_baum_welch(X, update_transitions=True)
    models = {}
    for name, model in [('supervised', supervised), ('baum_welch', refined), ('baum_welch_free_transitions', free)]:
        predictions = [model.predict_viterbi(s) for s in X]
        mapping, counts = state_mapping(y, predictions, model.n_states)
        models[name] = {'parameters': parameters(model), 'state_to_label': mapping.tolist(),
                        'mapping_train_confusion': counts.tolist(),
                        'mapping_training_correct': int(counts[mapping, np.arange(model.n_states)].sum()),
                        'training_log_likelihood': model.score(X)}
    for name, model in [('baum_welch', refined), ('baum_welch_free_transitions', free)]:
        models[name].update(history=model.history_, invariant_history=model.invariant_history_,
                            converged=model.converged_, stop_reason=model.stop_reason_, iterations=model.n_iter_,
                            update_transitions=model.update_transitions_)
    result = {'protocol': PROTOCOL, 'state_order': config['state_order'], 'feature_order': config['feature_order'],
              'fit_sequences': [s['name'] for s in train], 'fit_windows': sum(len(s['X']) for s in train),
              'training_feature_sha256': {s['name']: file_hash(root/'data/features/train'/f"{s['name']}.csv") for s in train},
              'feature_transform_sha256': file_hash(root/'data/features/transform.json'),
              'split_manifest_sha256': file_hash(root/'data/split_manifest.csv'),
              'source_sha256': {name: file_hash(Path(__file__).with_name(name)) for name in ['hmm.py', 'features.py', 'train_activity.py']},
              'supervised_counts': {key: value.tolist() for key, value in supervised.supervised_counts_.items()},
              'models': models}
    output = root/'data/models/activity_models.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def load_models(root):
    """Load frozen model artifacts and reject stale training/code/transform provenance."""
    root = Path(root)
    result = json.loads((root/'data/models/activity_models.json').read_text())
    if result['protocol'] != PROTOCOL:
        raise ValueError("Training protocol changed; retrain before evaluation")
    for name, digest in result['source_sha256'].items():
        if file_hash(Path(__file__).with_name(name)) != digest:
            raise ValueError("Model source changed; retrain before evaluation")
    for name, key in [('features/transform.json', 'feature_transform_sha256'), ('split_manifest.csv', 'split_manifest_sha256')]:
        if file_hash(root/'data'/name) != result[key]:
            raise ValueError("Feature transform or split changed; retrain")
    train = load_feature_sequences(root, 'train')
    if {s['name'] for s in train} != set(result['fit_sequences']):
        raise ValueError("Training membership changed")
    for name, digest in result['training_feature_sha256'].items():
        if file_hash(root/'data/features/train'/f'{name}.csv') != digest:
            raise ValueError("Training features changed; retrain")
    for record in result['models'].values():
        model_from_parameters(record, result['protocol'])
        if sorted(record['state_to_label']) != list(range(PROTOCOL['n_states'])):
            raise ValueError("Invalid frozen state mapping")
    return result


if __name__ == '__main__':
    result = train_models(Path(__file__).resolve().parents[1])
    em = result['models']['baum_welch']
    print(json.dumps({'fit_sequences': len(result['fit_sequences']), 'fit_windows': result['fit_windows'],
                      'em_iterations': em['iterations'], 'em_stop_reason': em['stop_reason'],
                      'initial_LL': em['history'][0], 'final_LL': em['history'][-1],
                      'state_mappings': {k: v['state_to_label'] for k,v in result['models'].items()}}, indent=2))
