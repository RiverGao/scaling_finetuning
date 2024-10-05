import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import sys


view = 'number'
model_residue = True
data_residue = True
head_pool = 'mean'

# view = sys.argv[1]
# model_residue = eval(sys.argv[2])
# data_residue = eval(sys.argv[3])
# head_pool = sys.argv[4]

d_size_layer = {'7B': 32, '13B': 40}
d_noise_ceiling = {
    (1, False, 'number'): 0.38548,
    (1, False, 'duration'): 0.36793,
    (1, True, 'number'): 0.33648,
    (1, True, 'duration'): 0.32455,
    (2, False, 'number'): 0.48524,
    (2, False, 'duration'): 0.45583,
    (2, True, 'number'): 0.41762,
    (2, True, 'duration'): 0.40640
}

font = {'family': 'DejaVu Sans',
        'weight': 'normal',
        'size': 25}
matplotlib.rc('font', **font)
model_prefix = 'residue_' if model_residue else ''
data_prefix = 'residue_' if data_residue else ''


def operate(model_size, lang_group):
    # l_group: 1 or 2
    # plot the saccade correlation with llama, alpaca and vicuna within a same model size
    # compare different models layer by layer
    n_layer = d_size_layer[model_size]

    corr_mean = [[], [], []]  # mean correlation per layer for the three models
    corr_error = [[], [], []]  # mean standard error

    for layer in range(n_layer):
        corrs = []  # [[llama corrs], [alpaca corrs], [vicuna corrs]]
        for model in ['llama', 'alpaca-lora', 'vicuna']:
            df_corr_layer = pd.read_excel(
                f'../../results/correlation/{model}/{model_size}/{head_pool}_{model_prefix}m_{data_prefix}s_sac_p{lang_group}_pearson.xlsx',
                sheet_name=f'layer {layer}')
            # print(df_corr_layer.columns)  # ['subject', 'r number', 'r duration']
            corrs.append(df_corr_layer[f'r {view}'].to_numpy())

        for ic, cor in enumerate(corrs):
            assert (lang_group == 1 and len(cor) == 51) or (lang_group == 2 and len(cor) == 54)  # number of subjects
            # calculate the mean and error of layer-wise correlations
            corr_mean[ic].append(np.mean(cor))
            corr_error[ic].append(stats.sem(cor))

    noise_ceiling = d_noise_ceiling[(lang_group, data_residue, view)]
    print(f'{view}, {model_size}, {lgroup}')
    print(f'Max Corr: {np.max(corr_mean[0])} in layer {np.argmax(corr_mean[0])} for llama, {np.max(corr_mean[1])} in layer {np.argmax(corr_mean[1])} for alpaca, {np.max(corr_mean[2])} in layer {np.argmax(corr_mean[2])} for vicuna')

    plot(
        n_layer,
        ['LLaMA', 'Alpaca', 'Vicuna', 'Noise Ceiling'],
        corr_mean,
        corr_error,
        noise_ceiling,
        f'{"Residual " if data_residue else ""}Saccade {view.capitalize()} vs. {"Residual " if model_residue else ""}{model_size} Models (L{lang_group}, {head_pool.capitalize()} Heads)',
        f'{model_prefix}m_vs_{data_prefix}s/{head_pool}_{model_prefix}m_vs_{data_prefix}s_sac-{view}-{lang_group}-{model_size}'
    )


def plot(n_ticks, line_labels, means, errors, noise, title, filename):
    # each plot has 4 lines, of which one is noise
    x = np.arange(n_ticks)  # the label locations
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ['black', 'xkcd:soft blue', 'xkcd:reddish', 'tab:gray']
    for il in range(3):
        # 4 lines, the last is supposed to be noise
        line = ax.errorbar(x, means[il], yerr=errors[il], label=line_labels[il], color=colors[il], ecolor='tab:gray')
    # assert line_labels[3] == 'Noise Ceiling'
    # line = ax.axhline(noise, x[0], x[-1], label=line_labels[3], color=colors[3])
    # only a horizontal line

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
    fig.savefig(f'../../results/figs/reading_brain/correlation/saccade/{filename}(no-noise).png', dpi=80)
    plt.close(fig)


if __name__ == '__main__':
    for msize in ['7B', '13B']:
        for lgroup in [1, 2]:
            operate(msize, lgroup)
