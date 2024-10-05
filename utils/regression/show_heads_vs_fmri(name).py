import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

p = 2
model_size = '13B'
alter = 'greater'
region = 'anterior'

d_size_layers = {'large': 36, '7B': 32, '13B': 40, '30B': 60}

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
n_layers = d_size_layers[model_size]


models = [f'llama_{model_size}', f'alpaca_{model_size}', f'vicuna_{model_size}']
names = ['LLaMA', 'Alpaca', 'Vicuna']
# models = [f'llama_{model_size}']
# names = ['LLaMA']
colors = ['black', 'xkcd:soft blue', 'xkcd:reddish', 'tab:gray']
fig, ax = plt.subplots(figsize=(14, 7))

llama_lr_scores = None
x = np.arange(n_layers)
noise = ceiling_dicts[p][region]

for i, model in enumerate(models):
    lr_scores = np.load(
        f'../../results/lr_scores_reading_brain/heads_vs_fmri/num_{p}_{region}_{model}.npy')  # layers, subjects
    if i == 0:
        llama_lr_scores = lr_scores
    means = np.mean(lr_scores, axis=1)  # (layers,)
    errors = [stats.sem(s) for s in lr_scores]  # layers,

    line = ax.errorbar(x, means, yerr=errors, label=names[i], color=colors[i], ecolor=colors[-1])
    print(f'Max score / ceiling of {model} is {np.max(means) / noise * 100:.2f}')

    # test whether fine-tuned models are different from pretrained
    if i > 0:  # alpaca and vicuna
        for layer in range(n_layers):
            t_value, p_value = stats.ttest_rel(lr_scores[layer], llama_lr_scores[layer], alternative=alter)
            if p_value < 0.05 / (4 * n_layers):
                print(f'{model} layer {layer} is significantly {alter} than llama, p={p_value}')

line = ax.axhline(noise, 0, x[-1], label='Noise Ceiling', color=colors[-1])

ax.set_xlabel('Model Layers')
ax.set_ylabel('Mean LR Scores')
ax.set_title(f'Regression Scores of {model_size} Model Attention vs. Saccade (L{p})')

ax.set_xticks([0, n_layers], ['Layer 0', f'Layer {n_layers - 1}'])
# ax.set_yticks([0, 0.025, 0.05])
ax.set_ylim((0, noise + 0.01))

ax.spines[['right', 'top']].set_visible(False)
for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2)
ax.tick_params(width=2)

ax.legend(frameon=False)
fig.tight_layout()

# plt.show()
fig.savefig(f'../../results/figs/reading_brain/lr-scores/heads_vs_fmri_L{p}_{model_size}(name).png', dpi=80)
plt.close(fig)



