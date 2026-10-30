"""Phase 7 parameter and outcome figures from frozen model/evaluation artifacts."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __package__:
    from .train_activity import load_models, file_hash
else:
    from train_activity import load_models, file_hash

SHORT_FEATURES = ['mean ax', 'mean ay', 'std |a|', 'var |ω|', 'SMA a', 'SMA ω',
                  'corr y,z', 'peak Hz', 'band ratio', 'entropy']


def aligned_parameters(frozen, name):
    """Reorder both transition axes and emission rows into frozen activity order."""
    record = frozen['models'][name]
    order = np.argsort(record['state_to_label'])
    values = {key: np.asarray(value) for key, value in record['parameters'].items()}
    return {'A': values['A'][np.ix_(order, order)], 'pi': values['pi'][order],
            'means': values['means'][order], 'variances': values['variances'][order]}


def plot_parameters(frozen):
    """Figure 2: primary model's transition probabilities and Gaussian mean z-scores."""
    values = aligned_parameters(frozen, frozen['protocol']['primary_model'])
    labels = [label.title() for label in frozen['state_order']]
    with plt.rc_context({'font.size': 12}):
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.8), layout='constrained')
        sns.heatmap(values['A'], annot=True, fmt='.3f', vmin=0, vmax=1, cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=axes[0], cbar_kws={'label': 'Probability'})
        axes[0].set(title='(a) Baum–Welch transition matrix', xlabel='Next state', ylabel='Current state')
        limit = max(1.0, float(np.max(abs(values['means']))))
        sns.heatmap(values['means'], cmap='vlag', center=0, vmin=-limit, vmax=limit,
                    xticklabels=SHORT_FEATURES, yticklabels=labels, ax=axes[1], cbar_kws={'label': 'Mean z-score'})
        axes[1].set(title='(b) Diagonal Gaussian emission means', xlabel='Feature (training-standardised)', ylabel='State')
        axes[1].tick_params(axis='x', labelrotation=35, labelsize=11)
        for ax in axes:
            ax.tick_params(axis='y', labelrotation=0)
        return fig


def plot_transition_comparison(frozen):
    """Notebook-only before/after transition matrices, aligned using training maps."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout='constrained')
    for ax, name in zip(axes, ['supervised', 'baum_welch']):
        sns.heatmap(aligned_parameters(frozen, name)['A'], annot=True, fmt='.3f',
                    vmin=0, vmax=1, cmap='Blues', xticklabels=frozen['state_order'],
                    yticklabels=frozen['state_order'], cbar=False, ax=ax)
        ax.set(title=name.replace('_', ' ').title(), xlabel='Next state', ylabel='Current state')
        ax.tick_params(axis='y', labelrotation=0)
    return fig


def plot_gaussians(frozen):
    """Notebook-only univariate marginals for three preselected interpretable features."""
    values = aligned_parameters(frozen, frozen['protocol']['primary_model'])
    chosen = ['mean_acc_y', 'std_acc_mag', 'dominant_freq_acc']
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), layout='constrained')
    for ax, name in zip(axes, chosen):
        j = frozen['feature_order'].index(name)
        means, std = values['means'][:, j], np.sqrt(values['variances'][:, j])
        grid = np.linspace(np.min(means-4*std), np.max(means+4*std), 1500)
        for i, label in enumerate(frozen['state_order']):
            density = np.exp(-.5*((grid-means[i])/std[i])**2)/(std[i]*np.sqrt(2*np.pi))
            ax.plot(grid, density, label=label)
        ax.set(title=SHORT_FEATURES[j], xlabel='Training-standardised z-score', ylabel='Marginal density')
    axes[-1].legend(fontsize=8)
    return fig


def plot_confusion_and_convergence(frozen, confusion):
    """Figure 3: fixed primary test confusion and training-only EM likelihood history."""
    name = frozen['protocol']['primary_model']
    history = frozen['models'][name]['history']
    with plt.rc_context({'font.size': 12}):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout='constrained')
        sns.heatmap(np.array(confusion[name]['test']), annot=True, fmt='d', cmap='Blues',
                    xticklabels=[s.title() for s in frozen['state_order']],
                    yticklabels=[s.title() for s in frozen['state_order']], cbar=False, ax=axes[0])
        axes[0].set(title='(a) Held-out windows (N=65)', xlabel='Predicted activity', ylabel='True activity')
        axes[0].tick_params(axis='x', labelrotation=30)
        axes[0].tick_params(axis='y', labelrotation=0)
        axes[1].plot(range(len(history)), history, marker='o', color='C1')
        axes[1].set(title='(b) Training convergence', xlabel='Accepted EM updates (0 = initial)',
                    ylabel='Total training log likelihood')
        axes[1].grid(alpha=.3)
        return fig


def plot_timelines(root, frozen, predictions, names):
    """Independent clip timelines: labels at window centres, processed |a| below.

    Each prediction covers a 2 s window; plotted centre-to-centre steps show
    decisions every 1 s and must not be interpreted as new ground-truth events.
    No gap, clip or missing tail is joined to manufacture a transition sequence.
    """
    root = Path(root)
    labels = frozen['state_order']
    rows = (len(names)+1)//2
    with plt.rc_context({'font.size': 11}):
        fig = plt.figure(figsize=(9, 3.05*rows), layout='constrained')
        outer = fig.add_gridspec(rows, 2)
        legend_handles = None
        for index, name in enumerate(names):
            part = predictions[predictions.sequence_id == name].sort_values('window_index')
            if part.empty:
                raise ValueError(f'No predictions for {name}')
            processed = pd.read_csv(root/'data/processed/test'/f'{name}.csv')
            offset = float(processed.seconds_elapsed.iloc[0])
            centres = (part.start_s.to_numpy()+part.stop_s.to_numpy())/2-offset
            nested = outer[index//2, index%2].subgridspec(2, 1, height_ratios=[1.2, 1])
            upper = fig.add_subplot(nested[0])
            lower = fig.add_subplot(nested[1], sharex=upper)
            upper.step(centres, part.true_state, where='mid', color='C0', linewidth=3, label='True')
            upper.step(centres, part.predicted_state, where='mid', color='C3', linewidth=1.6,
                       linestyle='--', marker='.', markersize=7, label='Predicted')
            upper.set(yticks=range(len(labels)), yticklabels=[s.title() for s in labels],
                      ylim=(-.4, len(labels)-.6), title=name)
            upper.tick_params(axis='x', labelbottom=False)
            upper.grid(axis='y', alpha=.25)
            if legend_handles is None:
                legend_handles = upper.get_legend_handles_labels()
            magnitude = np.linalg.norm(processed[['acc_x', 'acc_y', 'acc_z']].to_numpy(), axis=1)
            lower.plot(processed.seconds_elapsed-offset, magnitude, color='0.25', linewidth=1)
            lower.set(xlabel='Seconds since retained clip start', ylabel='|a| (m/s²)')
            lower.grid(alpha=.2)
        fig.legend(*legend_handles, loc='upper center', bbox_to_anchor=(.5, 1.05), ncol=2)
        return fig


def generate_figures(root):
    """Export report Figures 2–4 at 180 dpi from the fixed primary model/results."""
    root = Path(root)
    frozen = load_models(root)
    summary = json.loads((root/'data/evaluation/summary.json').read_text())
    if summary['model_artifact_sha256'] != file_hash(root/'data/models/activity_models.json'):
        raise ValueError('Evaluation belongs to a different model artifact')
    if summary['evaluation_source_sha256'] != file_hash(Path(__file__).with_name('evaluate_activity.py')):
        raise ValueError('Evaluation code changed; regenerate results before plotting')
    name = frozen['protocol']['primary_model']
    predictions = pd.read_csv(root/'data/evaluation/predictions'/f'{name}_test.csv')
    confusion = json.loads((root/'data/evaluation/confusion_matrices.json').read_text())
    # The representative selection is fixed by activity/name, never by accuracy.
    manifest = pd.read_csv(root/'data/split_manifest.csv')
    names = [sorted(manifest.loc[(manifest.split=='test') & (manifest.activity==label), 'file'])[0]
             for label in frozen['state_order']]
    figures = {'fig2_transition_emissions.png': plot_parameters(frozen),
               'fig3_confusion_convergence.png': plot_confusion_and_convergence(frozen, confusion),
               'fig4_test_timelines.png': plot_timelines(root, frozen, predictions, names)}
    for filename, fig in figures.items():
        fig.savefig(root/'figures'/filename, dpi=180, bbox_inches='tight')
    return figures


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    figures = generate_figures(root)
    for filename, fig in figures.items():
        print(root/'figures'/filename)
        plt.close(fig)
