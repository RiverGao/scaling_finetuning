import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
import matplotlib.ticker as mtick

view = 'num'
p = 2
region = 'anterior'

ceiling_dicts = {
    1: {'middle': 0.1422154901638094,
        'inferior': 0.14655120369083222,
        'superior': 0.13922533179913824,
        'angular': 0.15382562383967927,
        'anterior': 0.1429371952557311,
        },
    2: {'middle': 0.13910987315438422,
        'inferior': 0.1404541365418864,
        'superior': 0.12844558073711515,
        'angular': 0.14174022251211715,
        'anterior': 0.13334602065874473,
        },
}

font = {'family': 'DejaVu Sans',
        'weight': 'normal',
        'size': 20}
matplotlib.rc('font', **font)
noise = ceiling_dicts[p][region]

models = ['gpt2_large', 'llama_7B', 'llama_13B', 'llama_30B', 'llama_65B']
names = ['GPT2-774M', 'LLaMA-7B', 'LLaMA-13B', 'LLaMA-30B', 'LLaMA-65B']
colors = ['green', 'xkcd:soft blue', 'xkcd:reddish', 'purple', 'black', 'tab:gray']
markers = ['o', 'v', '>', '^', 'x']
fig, ax = plt.subplots(figsize=(14, 7))

for i, model in enumerate(models):
    lr_scores = np.load(
        f'../../results/lr_scores_reading_brain/heads_vs_fmri/num_{p}_{region}_{model}.npy')  # layers, subjects
    lr_scores = lr_scores / noise
    means = np.mean(lr_scores, axis=1)  # (layers,)
    errors = [stats.sem(s) for s in lr_scores]  # layers,

    x = np.arange(len(lr_scores))
    line = ax.errorbar(
        x, means, yerr=errors, label=names[i],
        linewidth=3, marker=markers[i],
        elinewidth=2.5,
        color=colors[i], ecolor='tab:gray',
    )
    print(f'Max score / ceiling of {model}: {np.max(means) * 100 :.2f}')

# line = ax.axhline(noise, 0, 59, label='Noise Ceiling', linewidth=3, color=colors[-1])

ax.set_xlabel('Model Layers')
ax.set_ylabel('Mean Human Resemblance')
ax.set_title(f'Layerwise L{p} Human Resemblance of Models in Different Scales')

ax.set_xticks([0, 79], ['Layer 0', 'Layer 79'])
# ax.set_yticks([0, 0.025, 0.05])
# ax.set_ylim((-0.005, 0.06))

ax.spines[['right', 'top']].set_visible(False)
for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2)
ax.tick_params(width=2)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))

ax.legend(frameon=False)
fig.tight_layout()

# plt.show()
fig.savefig(f'../../results/figs/reading_brain/lr-scores/fmri_{region}_resemb-size-L{p}.png', dpi=120)
plt.close(fig)
