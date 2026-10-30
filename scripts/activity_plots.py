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
