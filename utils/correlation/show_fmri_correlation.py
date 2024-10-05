import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import sys

model_residue = eval(sys.argv[1])
data_residue = eval(sys.argv[2])
head_pool = sys.argv[3]
region = sys.argv[4]

# model_residue = False
# data_residue = True
# head_pool = 'mean'
# region = 'angular'

# L1, residue False, region angular: 0.15696216551857997
# L1, residue True, region angular: 0.15407996573207122
# L1, residue False, region middle: 0.14316842781514283
# L1, residue True, region middle: 0.13908396494631547
# L1, residue False, region superior: 0.14081930620790814
# L1, residue True, region superior: 0.13770228226259545
# L1, residue False, region inferior: 0.14914600592753058
# L1, residue True, region inferior: 0.14472193473430056
# L1, residue False, region anterior: 0.1429371952557311
# L1, residue True, region anterior: 0.13938128784383105
# L2, residue False, region angular: 0.14174022251211715
# L2, residue True, region angular: 0.14212557983009744
# L2, residue False, region middle: 0.13910987315438422
# L2, residue True, region middle: 0.13662979372892434
# L2, residue False, region superior: 0.12844558073711515
# L2, residue True, region superior: 0.12669376068221122
# L2, residue False, region inferior: 0.1404541365418864
# L2, residue True, region inferior: 0.1406713830446618
# L2, residue False, region anterior: 0.13334602065874473
# L2, residue True, region anterior: 0.13189975651509056
d_size_layer = {'7B': 32, '13B': 40}
d_noise_ceiling = {
    (1, False, 'angular'): 0.15696216551857997,
    (1, True, 'angular'): 0.15407996573207122,
    (1, False, 'middle'): 0.1431684278151428,
    (1, True, 'middle'): 0.13908396494631547,
    (1, False, 'superior'): 0.14081930620790814,
    (1, True, 'superior'): 0.13770228226259545,
    (1, False, 'inferior'): 0.14914600592753058,
    (1, True, 'inferior'): 0.14472193473430056,
    (1, False, 'anterior'): 0.1429371952557311,
    (1, True, 'anterior'): 0.13938128784383105,
    (2, False, 'angular'): 0.14174022251211715,
    (2, True, 'angular'): 0.14212557983009744,
    (2, False, 'middle'): 0.13910987315438422,
    (2, True, 'middle'): 0.13662979372892434,
    (2, False, 'superior'): 0.12844558073711515,
    (2, True, 'superior'): 0.12669376068221122,
    (2, False, 'inferior'): 0.1404541365418864,
    (2, True, 'inferior'): 0.1406713830446618,
    (2, False, 'anterior'): 0.13334602065874473,
    (2, True, 'anterior'): 0.13189975651509056,
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
                f'../../results/correlation/{model}/{model_size}/{head_pool}_{model_prefix}m_{data_prefix}f_fmri_{region}_p{lang_group}_pearson.xlsx',
                sheet_name=f'layer {layer}')
            # print(df_corr_layer.columns)  # ['subject', 'r number', 'r duration']
            corrs.append(df_corr_layer[f'r activation'].to_numpy())

        for ic, cor in enumerate(corrs):
            # assert (lang_group == 1 and len(cor) == 51) or (lang_group == 2 and len(cor) == 54)  # number of subjects
            # calculate the mean and error of layer-wise correlations
            corr_mean[ic].append(np.mean(cor).__abs__())
            corr_error[ic].append(stats.sem(cor))

    noise_ceiling = d_noise_ceiling[(lang_group, data_residue, region)]

    plot(
        n_layer,
        ['LLaMA', 'Alpaca', 'Vicuna', 'Noise Ceiling'],
        corr_mean,
        corr_error,
        noise_ceiling,
        f'{"Residue " if data_residue else ""}{region.capitalize()} vs. {"Residue " if model_residue else ""}{model_size} Models (L{lang_group}, {head_pool.capitalize()} Heads)',
        f'{model_size}/{region}/L{lang_group}/{head_pool}_{model_prefix}m_{data_prefix}f_fmri-{region}-{lang_group}-{model_size}'
    )


def plot(n_ticks, line_labels, means, errors, noise, title, filename):
    # each plot has 4 lines, of which one is noise
    x = np.arange(n_ticks)  # the label locations
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ['black', 'xkcd:soft blue', 'xkcd:reddish', 'tab:gray']
    for il in range(3):
        # 4 lines, the last is supposed to be noise
        line = ax.errorbar(x, means[il], yerr=errors[il], label=line_labels[il], color=colors[il], ecolor='tab:gray')
    assert line_labels[3] == 'Noise Ceiling'
    line = ax.axhline(noise, x[0], x[-1], label=line_labels[3], color=colors[3])
    # only a horizontal line

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Model Layers')
    ax.set_ylabel(f"Mean Absolute Pearson's Correlation")
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
    fig.savefig(f'../../results/figs/reading_brain/correlation/fmri/{filename}.png', dpi=80)
    plt.close(fig)


if __name__ == '__main__':
    for msize in ['7B', '13B']:
        for lgroup in [1, 2]:
            operate(msize, lgroup)
