import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import sys

method = 'js'
file_prefix = 'ctrl_'
attn_method = ''

d_size_layer = {'7B': 32, '13B': 40}

font = {'family': 'DejaVu Sans',
        'weight': 'normal',
        'size': 25}
matplotlib.rc('font', **font)


def operate_name(model_size):
    # plot the divergence between llama, alpaca and vicuna within a same model size
    # compare different models layer by layer
    n_layer = d_size_layer[model_size]

    div_mean = [[], [], [], []]  # mean divergence per layer for the three combinations and noise
    div_error = [[], [], [], []]  # mean standard error
    significant_t = [[], [], []]  # t values of rel-t test of the three combinations to noise
    significant_p = [[], [], []]  # p values
    count_ttest = 0

    for layer in range(n_layer):
        df_div_layer = pd.read_excel(
            f'../../results/divergence/instruct/{model_size}/{attn_method + "_" if attn_method else ""}{file_prefix}js.xlsx',
            sheet_name=f'layer {layer}')
        # print(df_div_layer.columns)  # ['llama-alpaca', 'llama-vicuna', 'alpaca-vicuna', 'llama-noise']

        # ahd stands for average head divergence
        ahds = [
            df_div_layer['llama'].to_numpy(),
            df_div_layer['alpaca'].to_numpy(),
            df_div_layer['vicuna'].to_numpy(),
            df_div_layer['noise'].to_numpy()
        ]

        for ia, ahd in enumerate(ahds):
            assert len(ahd) == 148

            # calculate the mean and error of layer-wise divergences
            div_mean[ia].append(np.mean(ahd))
            div_error[ia].append(stats.sem(ahd))

            if ia == 3:
                break

            # do rel-t tests between combinations and the noise
            t_div, p_div = stats.ttest_rel(ahd, ahds[3], alternative='greater')
            count_ttest += 1
            significant_t[ia].append(t_div)
            significant_p[ia].append(p_div)

    p_threshold = 0.05 / count_ttest
    print(f'Significant threshold is {p_threshold}')
    print(np.where(np.array(significant_p) < p_threshold))
    # plot(
    #     n_layer,
    #     ['LLaMA', 'Alpaca', 'Vicuna', 'Ref.'],
    #     div_mean,
    #     div_error,
    #     f'Attn. Div. between Plain and Instructed Text ({model_size})',
    #     f'div-instruct-{model_size}'
    # )
    plot(
        n_layer,
        ['LLaMA', 'Alpaca', 'Vicuna', 'Ref.'],
        div_mean,
        div_error,
        f'Attn. Div. between Plain and Noise-Prefixed Text ({model_size})',
        f'div-instruct-{model_size}'
    )


def plot(n_ticks, line_labels, means, errors, title, filename):
    # each plot has 4 lines, of which one is noise
    x = np.arange(n_ticks)  # the label locations
    fig, ax = plt.subplots(figsize=(14, 7))
    styles = ['dotted', 'dashed', 'dashdot', 'solid']
    colors = ['black', 'xkcd:soft blue', 'xkcd:reddish', 'tab:gray']
    for il in range(len(means)):
        # 4 lines, the last is supposed to be noise
        line = ax.errorbar(
            x, means[il], yerr=errors[il], label=line_labels[il],
            linewidth=3, linestyle=styles[il],
            elinewidth=2.5,
            color=colors[il], ecolor='tab:gray'
        )
        print(f'{line_labels[il]} mean: {np.mean(means[il])}, max: {np.max(means[il])}')
    # assert line_labels[2] == 'Noise'
    # line = ax.errorbar(x, means[2], label=line_labels[2], color=colors[2])  # do not show error bar of noise

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(f"Mean JS Divergence")
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
    fig.savefig(f'../../results/figs/reading_brain/divergence/{attn_method + "_" if attn_method else ""}{filename}.png', dpi=120)
    plt.close(fig)


if __name__ == '__main__':
    operate_name('7B')
    operate_name('13B')
