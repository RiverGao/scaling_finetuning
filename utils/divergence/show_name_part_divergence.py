import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import sys


residue = False
mission = 'size'

# residue = eval(sys.argv[1])
# mission = sys.argv[2]
method = 'js'
file_prefix = 'residue_' if residue else ''
file_prefix = '' + file_prefix
attn_method = 'rollout'

d_size_layer = {'7B': 32, '13B': 40}

font = {'family': 'DejaVu Sans',
        'weight': 'normal',
        'size': 25}
matplotlib.rc('font', **font)


def operate_name(model_size):
    # plot the divergence between llama, alpaca and vicuna within a same model size
    # compare different models layer by layer
    n_layer = d_size_layer[model_size]

    div_mean = [[], [], []]  # mean divergence per layer for the three combinations and noise
    div_error = [[], [], []]  # mean standard error
    significant_t = [[], []]  # t values of rel-t test of the three combinations to noise
    significant_p = [[], []]  # p values
    count_ttest = 0

    for layer in range(n_layer):
        df_div_layer = pd.read_excel(
            f'../../results/divergence/name/{model_size}/{attn_method + "_" if attn_method else ""}{file_prefix}js-name.xlsx',
            sheet_name=f'layer {layer}')
        # print(df_div_layer.columns)  # ['llama-alpaca', 'llama-vicuna', 'alpaca-vicuna', 'llama-noise']

        # ahd stands for average head divergence
        ahds = [
            df_div_layer['llama-alpaca'].to_numpy(),
            df_div_layer['llama-vicuna'].to_numpy(),
            # df_div_layer['alpaca-vicuna'].to_numpy(),
            df_div_layer['noise'].to_numpy()
        ]

        for ia, ahd in enumerate(ahds):
            assert len(ahd) == 148

            # calculate the mean and error of layer-wise divergences
            div_mean[ia].append(np.mean(ahd))
            div_error[ia].append(stats.sem(ahd))
            if ia == 2:
                break  # t test only for the 3 combinations

            # do rel-t tests between combinations and the noise
            t_div, p_div = stats.ttest_rel(ahd, ahds[2], alternative='greater')
            count_ttest += 1
            significant_t[ia].append(t_div)
            significant_p[ia].append(p_div)

    p_threshold = 0.05 / count_ttest
    print(f'Significant threshold is {p_threshold}')
    print(np.where(np.array(significant_p) < p_threshold))
    plot(
        n_layer,
        # ['LLaMA-Alpaca', 'LLaMA-Vicuna', 'Alpaca-Vicuna', 'Noise'],
        ['Alpaca-LLaMA', 'Vicuna-LLaMA', 'Ref.'],
        div_mean,
        div_error,
        f'{"Residue " if residue else ""} Attn. Div. between Pretrained and Tuned Models ({model_size})',
        f'div-name-{model_size}'
    )


def operate_size():
    # plot the divergence between 7B and 13B in llama, alpaca and vicuna
    # compare different sizes part by part
    div_mean = [[], [], [], []]  # mean divergence per part for the three models and noise
    div_error = [[], [], [], []]  # mean standard error
    significant_t = [[], [], []]  # t values of rel-t test of the three models to noise
    significant_p = [[], [], []]  # p values
    count_ttest = 0

    for part in range(4):
        df_div_part = pd.read_excel(
            f'../../results/divergence/size/{attn_method + "_" if attn_method else ""}{file_prefix}js-size.xlsx',
            sheet_name=f'part {part}')
        # print(df_div_part.columns)  # ['llama', 'alpaca', 'vicuna', 'noise']

        # ahd stands for average head divergence
        ahds = [df_div_part['llama'].to_numpy(),
                df_div_part['alpaca'].to_numpy(),
                df_div_part['vicuna'].to_numpy(),
                df_div_part['noise'].to_numpy()]

        for ia, ahd in enumerate(ahds):
            assert len(ahd) == 148

            # calculate the mean and error of part-wise divergences
            div_mean[ia].append(np.mean(ahd))
            div_error[ia].append(stats.sem(ahd))
            if ia == 3:
                break  # t test only for the 3 combinations

            # do rel-t tests between combinations and the noise
            t_div, p_div = stats.ttest_rel(ahd, ahds[3], alternative='greater')
            count_ttest += 1
            significant_t[ia].append(t_div)
            significant_p[ia].append(p_div)

    p_threshold = 0.05 / count_ttest
    print(f'Significant threshold is {p_threshold}')
    print(np.where(np.array(significant_p) < p_threshold))
    plot(
        4,
        ['LLaMA', 'Alpaca', 'Vicuna', 'Ref.'],
        div_mean,
        div_error,
        f'{"Residue " if residue else ""}Attention Divergence between 7B and 13B',
        'div-size'
    )


def plot(n_ticks, line_labels, means, errors, title, filename):
    # each plot has 4 lines, of which one is noise
    x = np.arange(n_ticks)  # the label locations
    fig, ax = plt.subplots(figsize=(14, 7))
    styles = ['dotted', 'dashed', 'dashdot', 'solid'] if mission == 'size' else ['dashed', 'dashdot', 'solid']
    colors = ['black', 'xkcd:soft blue', 'xkcd:reddish', 'tab:gray'] if mission == 'size' else ['xkcd:soft blue', 'xkcd:reddish', 'tab:gray']
    for il in range(len(means)):
        # 3 lines, the last is supposed to be noise
        line = ax.errorbar(
            x, means[il], yerr=errors[il], label=line_labels[il],
            linewidth=3, linestyle=styles[il],
            elinewidth=2.5,
            color=colors[il], ecolor='tab:gray'
        )
    # assert line_labels[2] == 'Noise'
    # line = ax.errorbar(x, means[2], label=line_labels[2], color=colors[2])  # do not show error bar of noise

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(f"Mean JS Divergence")
    ax.set_title(title)

    if n_ticks > 4:
        ax.set_xticks([0, n_ticks - 1], ['Layer 0', f'Layer {n_ticks - 1}'])
    else:
        ax.set_xticks([0, 1, 2, 3], ['25%', '50%', '75%', '100%'])
    # ax.set_yticks([0, 0.025, 0.05])
    # ax.set_ylim((-0.005, 0.06))

    ax.spines[['right', 'top']].set_visible(False)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(2)
    ax.tick_params(width=2)

    ax.legend(frameon=False)
    fig.tight_layout()

    # plt.show()
    fig.savefig(f'../../results/figs/reading_brain/divergence/{attn_method + "_" if attn_method else ""}{file_prefix}{filename}.png', dpi=120)
    plt.close(fig)


if __name__ == '__main__':
    if mission == 'name':
        operate_name('7B')
        operate_name('13B')
    elif mission == 'size':
        operate_size()
