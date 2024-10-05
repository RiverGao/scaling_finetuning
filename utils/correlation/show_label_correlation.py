import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import sys


residue = eval(sys.argv[1])
head_pool = sys.argv[2]

d_size_layer = {'7B': 32, '13B': 40}

font = {'family': 'DejaVu Sans',
        'weight': 'normal',
        'size': 25}
matplotlib.rc('font', **font)
file_prefix = 'residue_' if residue else ''
file_prefix = head_pool + '_' + file_prefix


def operate(model_size, label):
    # label: attribute, predicate, adverbial
    # plot the label correlation with llama, alpaca and vicuna within a same model size
    # compare different models layer by layer
    n_layer = d_size_layer[model_size]

    corr_mean = [[], [], []]  # mean correlation per layer for the three models
    corr_error = [[], [], []]  # mean standard error

    for layer in range(n_layer):
        corrs = []  # [[llama corrs], [alpaca corrs], [vicuna corrs]]
        for model in ['llama', 'alpaca-lora', 'vicuna']:
            df_corr_layer = pd.read_excel(
                f'../../results/correlation/{model}/{model_size}/{file_prefix}{label}_pearson.xlsx',
                sheet_name=f'layer {layer}')
            # print(df_corr_layer.columns)  # ['subject', 'r number', 'r duration']
            corrs.append(df_corr_layer[f'r {label}'].to_numpy())

        for ic, cor in enumerate(corrs):
            # assert len(cor) == 148  # number of sentences in all 5 articles
            # calculate the mean and error of layer-wise correlations
            corr_mean[ic].append(np.mean(cor))
            corr_error[ic].append(stats.sem(cor))

    print(f'{label}, {model_size}, {residue}')
    print(f'Max Corr: {np.max(corr_mean[0])} for llama, {np.max(corr_mean[1])} for alpaca, {np.max(corr_mean[2])} for vicuna')

    plot(
        n_layer,
        ['LLaMA', 'Alpaca', 'Vicuna'],
        corr_mean,
        corr_error,
        f'{"Residue " if residue else ""}{label.capitalize()} Correlation of {model_size} Models ({head_pool.capitalize()} Heads)',
        f'{file_prefix}label-{label}-{model_size}'
    )


def plot(n_ticks, line_labels, means, errors, title, filename):
    # each plot has 3 lines
    x = np.arange(n_ticks)  # the label locations
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ['black', 'xkcd:soft blue', 'xkcd:reddish']
    for il in range(3):
        # 3 lines
        line = ax.errorbar(x, means[il], yerr=errors[il], label=line_labels[il], color=colors[il], ecolor='tab:gray')
        # line = ax.plot(x, means[il], label=line_labels[il], color=colors[il])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Model Layers')
    ax.set_ylabel(f"Mean Pearson's Correlation")
    ax.set_title(title)

    ax.set_xticks([0, n_ticks - 1], ['Layer 0', f'Layer {n_ticks - 1}'])
    # ax.set_yticks([0, 0.025, 0.05])
    # ax.set_ylim((-0.005, 0.06))

    ax.spines[['right', 'top']].set_visible(False)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(2)
    ax.tick_params(width=2)

    ax.legend(frameon=False)
    fig.tight_layout()

    # plt.show()
    fig.savefig(f'../../results/figs/reading_brain/correlation/label/{filename}.png', dpi=80)
    plt.close(fig)


if __name__ == '__main__':
    for msize in ['7B', '13B']:
        for lbl in ['attribute', 'predicate']:
            operate(msize, lbl)
