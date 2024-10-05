import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats


task = 'predicate'
model_type = 'llama'
model_size = '30B'
model_layer = 9
method = 'pearson'
residue = True
head_pool = 'mean'

d_size_layer = {
    'gpt2-large': 20,
    'llama-7B': 32,
    'llama-13B': 40,
    'llama-30B': 52
}
if model_type in ['llama', 'alpaca', 'vicuna']:
    n_layers = d_size_layer['llama-' + model_size]
else:
    n_layers = d_size_layer[model_type + '-' + model_size]

file_prefix = 'residue_' if residue else ''

font = {'family': 'DejaVu Sans',
        'weight': 'normal',
        'size': 25}

matplotlib.rc('font', **font)

labels = [f'Layer {i}' for i in range(n_layers)]
r_amr_model_means = []
r_amr_model_error = []

for layer in range(n_layers):
    df_model = pd.read_excel(
        f'../../results/correlation/{model_type}/{model_size}/layer{model_layer}_{file_prefix}{task}_{method}.xlsx',
        sheet_name=f'head {layer}')

    r_amr_model = df_model[f'r {task}'].dropna().to_numpy()
    r_amr_model_means.append(np.mean(r_amr_model))
    r_amr_model_error.append(stats.sem(r_amr_model))


print(f'{task}, {residue}')
print(f'Max Corr: {np.max(r_amr_model_means)} in head {np.argmax(r_amr_model_means)} for {model_type} {model_size} layer {model_layer}')

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(14, 7))
rects1 = ax.bar(x - 0.5 * width, r_amr_model_means, width, yerr=r_amr_model_error,
                label=f'{task.capitalize()} Label vs. model attention', color='xkcd:soft blue', ecolor='tab:gray')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel(f"{model_type.upper()} Heads")
ax.set_ylabel("Absolute mean " + method.capitalize() + "'s " + r"$\it{r}$")
ax.set_title(f"{task.capitalize()} Label vs. {model_type.capitalize()} Attention ({model_size})")
ax.set_xticks([0, n_layers - 1], ['Head 1', f'Head {n_layers}'])
ax.set_yticks([0, 0.1, 0.2])
ax.set_ylim((-0.05, 0.25))

ax.spines[['right', 'top']].set_visible(False)
for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2)
ax.tick_params(width=2)

ax.legend(frameon=False)
fig.tight_layout()

plt.show()

