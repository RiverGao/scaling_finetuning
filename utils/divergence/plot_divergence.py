import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

model_type = 't5'
model_size = 'base'
part = 'decoder'
method = 'KL'

d_size_layer = {
    'gpt2-base': 12,
    'gpt2-large': 36,
    'gpt2-multi': 24,
    'bert-base': 12,
    'bert-large': 24,
    't5-base': 12,
    't5-large': 24
}
n_layers = d_size_layer[model_type + '-' + model_size]

font = {'family': 'DejaVu Sans',
        'weight': 'normal',
        'size': 25}

matplotlib.rc('font', **font)

labels = [f'Layer {i}' for i in range(n_layers)]
r_amr_model_means = []
r_amr_model_error = []
r_amr_random_means = []
r_amr_random_error = []

for layer in range(n_layers):
    if not part:
        df_model = pd.read_excel(
            f'divergence/{model_type}/{model_size}/amr_{model_size}_{method}.xlsx',
            sheet_name=f'layer {layer}')
        df_random = pd.read_excel(
            f'divergence/{model_type}/{model_size}-random/amr_{model_size}-random_{method}.xlsx',
            sheet_name=f'layer {layer}')
    else:
        df_model = pd.read_excel(
            f'divergence/{model_type}/{model_size}/{part}/amr_{model_size}_{method}.xlsx',
            sheet_name=f'layer {layer}')
        df_random = pd.read_excel(
            f'divergence/{model_type}/{model_size}-random/{part}/amr_{model_size}-random_{method}.xlsx',
            sheet_name=f'layer {layer}')

    r_amr_model = df_model['divergence'].to_numpy()
    r_amr_model_means.append(np.mean(r_amr_model))
    r_amr_model_error.append(stats.sem(r_amr_model))

    r_amr_random = df_random['divergence'].to_numpy()
    r_amr_random_means.append(np.mean(r_amr_random))
    r_amr_random_error.append(stats.sem(r_amr_random))

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(14, 7))
rects1 = ax.bar(x - 0.5 * width, r_amr_model_means, width, yerr=r_amr_model_error,
                label='AMR connections vs. model attention', color='xkcd:soft blue', ecolor='tab:gray')
rects2 = ax.bar(x + 0.5 * width, r_amr_random_means, width, yerr=r_amr_random_error,
                label='AMR connections vs. random', color='xkcd:reddish', ecolor='tab:gray')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel(f"{model_type.upper()} Layers")
ax.set_ylabel(f"Mean {method} divergence")
if not part:
    ax.set_title(f"AMR connection vs. {model_type.upper()} Attention ({model_size})")
else:
    ax.set_title(f"AMR connection vs. {model_type.upper()} {part.capitalize()} Attention ({model_size})")
ax.set_xticks([0, n_layers - 1], ['Layer 1', f'Layer {n_layers}'])
ax.set_yticks([0, 1, 2, 3, 4])
ax.set_ylim((-0.5, 4.5))

ax.spines[['right', 'top']].set_visible(False)
for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2)
ax.tick_params(width=2)

ax.legend(frameon=False)
fig.tight_layout()

plt.show()

