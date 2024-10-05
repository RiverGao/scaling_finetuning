import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

view = 'num'
p = 2
residue = False
instruct = False
model_size = '13B'
instruct_prefix = 'para_' if instruct else ''
alter = 'greater'
attn_method = 'rollout'

d_size_layers = {'large': 36, '7B': 32, '13B': 40, '30B': 60}
d_noise_ceiling = {
    (1, False, 'num'): 0.1559854152570265,
    (2, False, 'num'): 0.24632999658975888,
    (1, False, 'dur'): 0.13783828991106906,
    (2, False, 'dur'): 0.21455276206747248,
    (1, True, 'num'): 0.12004709688250854,
    (2, True, 'num'): 0.1802391293492779,
    (1, True, 'dur'): 0.11021495334933562,
    (2, True, 'dur'): 0.1655612299768772,
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
noise = d_noise_ceiling[(p, residue, view)]

for i, model in enumerate(models):
    lr_scores = np.load(
        f'../../results/lr_scores_reading_brain/heads_vs_saccade/{attn_method + "_" if attn_method else ""}{instruct_prefix}num_{p}_{model}.npy')  # layers, subjects
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
fig.savefig(f'../../results/figs/reading_brain/lr-scores/{attn_method + "_" if attn_method else ""}heads_vs_sac_{view}_L{p}_{model_size}(name).png', dpi=80)
plt.close(fig)



