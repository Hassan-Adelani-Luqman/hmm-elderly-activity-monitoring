"""Phase 7 fixed-model evaluation; test outputs never change fit or label mapping."""

from pathlib import Path
import json
import numpy as np
import pandas as pd

if __package__:
    from .features import load_feature_sequences
    from .train_activity import load_models, model_from_parameters, file_hash
else:
    from features import load_feature_sequences
    from train_activity import load_models, model_from_parameters, file_hash


def predict_gaussian(model, X):
    """Independent Gaussian MAP labels (T,) using supervised emissions and pi.

    The same supervised initial distribution is applied at every window. A,
    neighbouring observations and decoded labels have no effect on this baseline.
    Supervised emission state IDs already equal the fixed activity label IDs.
    """
    with np.errstate(divide='ignore'):
        scores = model._log_emission(X) + np.log(model.pi)
    return np.argmax(scores, axis=1).astype(np.int64)


def confusion_matrix(y_true, y_pred, n_states):
    """Count true-label rows and predicted-label columns using NumPy only."""
    true, predicted = np.asarray(y_true), np.asarray(y_pred)
    if (not isinstance(n_states, (int, np.integer)) or n_states < 1
            or true.ndim != 1 or true.shape != predicted.shape or not len(true)
            or not np.issubdtype(true.dtype, np.integer) or not np.issubdtype(predicted.dtype, np.integer)
            or (true < 0).any() or (predicted < 0).any() or (true >= n_states).any() or (predicted >= n_states).any()):
        raise ValueError("Expected aligned nonempty integer labels in [0,N)")
    counts = np.zeros((n_states, n_states), dtype=np.int64)
    np.add.at(counts, (true, predicted), 1)
    return counts


def metrics_from_confusion(matrix, labels):
    """Return metric records, including Overall; count/rate arithmetic is NumPy.

    Per-class accuracy=(TP+TN)/N is one-vs-rest. Overall accuracy=trace(C)/N;
    Overall sensitivity/specificity are macro averages of defined class rates.
    Undefined denominators yield NaN, never a misleading zero or perfect score.
    """
    C = np.asarray(matrix)
    if (C.shape != (len(labels), len(labels)) or not np.issubdtype(C.dtype, np.integer)
            or (C < 0).any() or C.sum() == 0):
        raise ValueError("Expected a nonnegative integer confusion matrix with observations")
    total = int(C.sum())
    support, predicted = C.sum(axis=1), C.sum(axis=0)
    TP = np.diag(C); FN = support-TP; FP = predicted-TP; TN = total-TP-FN-FP
    sensitivity = np.divide(TP, TP+FN, out=np.full(len(labels), np.nan), where=TP+FN > 0)
    specificity = np.divide(TN, TN+FP, out=np.full(len(labels), np.nan), where=TN+FP > 0)
    rows = [{'activity': label, 'n_windows': int(support[i]), 'TP': int(TP[i]), 'FN': int(FN[i]),
             'FP': int(FP[i]), 'TN': int(TN[i]), 'sensitivity': float(sensitivity[i]),
             'specificity': float(specificity[i]), 'accuracy': float((TP[i]+TN[i])/total)}
            for i, label in enumerate(labels)]
    rows.append({'activity': 'Overall', 'n_windows': total, 'TP': None, 'FN': None, 'FP': None, 'TN': None,
                 'sensitivity': float(np.nanmean(sensitivity)) if np.isfinite(sensitivity).any() else float('nan'),
                 'specificity': float(np.nanmean(specificity)) if np.isfinite(specificity).any() else float('nan'),
                 'accuracy': float(np.trace(C)/total)})
    return rows


def evaluate_models(root):
    """Evaluate saved models independently per clip; never fit or remap on test.

    Write predictions, matrices, class/clip metrics and a frozen-model comparison.
    Both train and test metrics are reported; train scores are resubstitution.
    The predetermined primary model is retained regardless of test performance.
    """
    root = Path(root)
    frozen = load_models(root)
    model_path = root/'data/models/activity_models.json'
    model_digest = file_hash(model_path)
    sequences = {split: load_feature_sequences(root, split) for split in ['train', 'test']}
    if len(sequences['test']) != 8 or sum(len(s['X']) for s in sequences['test']) != 65:
        raise ValueError("Expected the fixed eight-clip/65-window test set")
    labels = frozen['state_order']
    output = root/'data/evaluation'
    (output/'predictions').mkdir(parents=True, exist_ok=True)
    metrics, clip_rows, comparison, matrices, predictions = [], [], [], {}, {}
    for name in frozen['protocol']['models']:
        record = frozen['models']['supervised' if name == 'gaussian_baseline' else name]
        model = model_from_parameters(record, frozen['protocol'])
        mapping = np.arange(len(labels)) if name == 'gaussian_baseline' else np.array(record['state_to_label'])
        matrices[name], predictions[name] = {}, {}
        summary = {'model': name, 'primary': name == frozen['protocol']['primary_model']}
        for split, items in sequences.items():
            frames = []
            for item in items:
                raw = predict_gaussian(model, item['X']) if name == 'gaussian_baseline' else model.predict_viterbi(item['X'])
                predicted = mapping[raw]
                frame = item['windows'].copy()
                frame['true_state'], frame['raw_model_state'], frame['predicted_state'] = item['y'], raw, predicted
                frame['true_activity'] = np.array(labels)[item['y']]
                frame['predicted_activity'] = np.array(labels)[predicted]
                frames.append(frame)
                correct = int(np.sum(item['y'] == predicted))
                clip_rows.append({'model': name, 'split': split, 'sequence_id': item['name'],
                                  'activity': labels[int(item['y'][0])], 'n_windows': len(predicted),
                                  'correct': correct, 'errors': len(predicted)-correct, 'accuracy': correct/len(predicted)})
            frame = pd.concat(frames, ignore_index=True)
            frame.to_csv(output/'predictions'/f'{name}_{split}.csv', index=False)
            predictions[name][split] = frame
            C = confusion_matrix(frame.true_state.to_numpy(), frame.predicted_state.to_numpy(), len(labels))
            matrices[name][split] = C.tolist()
            rows = metrics_from_confusion(C, labels)
            metrics.extend({'model': name, 'split': split, **row} for row in rows)
            summary[f'{split}_windows'] = int(C.sum())
            summary[f'{split}_correct'] = int(np.trace(C))
            summary[f'{split}_accuracy'] = rows[-1]['accuracy']
            summary[f'{split}_macro_sensitivity'] = rows[-1]['sensitivity']
            summary[f'{split}_macro_specificity'] = rows[-1]['specificity']
            summary[f'{split}_mean_clip_accuracy'] = float(np.mean([r['accuracy'] for r in clip_rows if r['model']==name and r['split']==split]))
        comparison.append(summary)
    metric_frame, clip_frame, comparison_frame = pd.DataFrame(metrics), pd.DataFrame(clip_rows), pd.DataFrame(comparison)
    metric_frame.to_csv(output/'metrics.csv', index=False)
    clip_frame.to_csv(output/'per_clip.csv', index=False)
    comparison_frame.to_csv(output/'comparison.csv', index=False)
    (output/'confusion_matrices.json').write_text(json.dumps(matrices, indent=2)+'\n')
    summary = {'primary_model': frozen['protocol']['primary_model'], 'model_artifact_sha256': model_digest,
               'evaluation_source_sha256': file_hash(__file__), 'state_order': labels,
               'evaluation_policy': frozen['protocol']['selection_policy'],
               'metric_definitions': {'sensitivity': 'TP/(TP+FN)', 'specificity': 'TN/(TN+FP)',
                                      'class_accuracy': '(TP+TN)/N, one-vs-rest',
                                      'overall_accuracy': 'sum(diagonal confusion)/N',
                                      'overall_sensitivity_specificity': 'Macro average across defined class rates'},
               'test_sequences': [s['name'] for s in sequences['test']],
               'test_feature_sha256': {s['name']: file_hash(root/'data/features/test'/f"{s['name']}.csv") for s in sequences['test']},
               'limitations': ['Eight clips, two per activity; 65 overlapping windows are not independent trials',
                               'Same participant/device/collection conditions; test clips previously quality-inspected',
                               'No real cross-activity test transitions, unseen-day/place or elderly/fall validation',
                               'Training accuracy is resubstitution, not validation performance']}
    if file_hash(model_path) != model_digest:
        raise RuntimeError("Evaluation must not modify the frozen model artifact")
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    return {'models': frozen, 'summary': summary, 'metrics': metric_frame, 'per_clip': clip_frame,
            'comparison': comparison_frame, 'confusion': matrices, 'predictions': predictions}


if __name__ == '__main__':
    result = evaluate_models(Path(__file__).resolve().parents[1])
    print(result['comparison'].to_string(index=False))
    print(result['metrics'].query("model == 'baum_welch' and split == 'test'").to_string(index=False))
    print(result['per_clip'].query("model == 'baum_welch' and split == 'test'").to_string(index=False))
