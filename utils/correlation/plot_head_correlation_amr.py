import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

model_type = 'gpt2'
model_size = 'large'
model_layer = 14
method = 'pearson'
residue = True

d_size_heads = {
    'gpt2-large': 20,
    'llama-7B': 32,
    'llama-13B': 40,
    'llama-30B': 52
}

file_prefix = 'residue_' if residue else ''

if model_type in ['llama', 'alpaca-lora', 'vicuna']:
    n_heads = d_size_heads['llama-' + model_size]
else:
    n_heads = d_size_heads[model_type + '-' + model_size]

font = {'family': 'DejaVu Sans',
        'weight': 'normal',
        'size': 25}

matplotlib.rc('font', **font)

labels = [f'Head {i}' for i in range(n_heads)]
r_amr_gpt_means = []
r_amr_gpt_error = []

for head in range(n_heads):
    df_gpt = pd.read_excel(
        f'../../results/correlation/{model_type}/{model_size}/layer{model_layer}_{file_prefix}amr_{method}.xlsx',
        sheet_name=f'head {head}'
    )

    r_amr_gpt = df_gpt['r AMR'].to_numpy()
    r_amr_gpt_means.append(np.mean(r_amr_gpt))
    r_amr_gpt_error.append(stats.sem(r_amr_gpt))


print(f'AMR, residue {residue}')
print(f'Max Corr: {np.max(r_amr_gpt_means)} in head {np.argmax(r_amr_gpt_means)} for {model_type} {model_size} layer {model_layer}')
x = np.arange(len(labels))  # the label locations
width = 0.22  # the width of the bars

fig, ax = plt.subplots(figsize=(14, 7))
rects1 = ax.bar(x - 1.5 * width, r_amr_gpt_means, width, yerr=r_amr_gpt_error,
                label=f'AMR vs. {model_type} ({model_size})', color='xkcd:soft blue', ecolor='tab:gray')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel(f"{model_type.upper()}-{model_size.capitalize()} Layers")
ax.set_ylabel(f"Mean {method.capitalize()}'s" + r"$\it{r}$")
ax.set_title(f"AMR vs. {model_type.upper()} Attention")
ax.set_xticks([0, n_heads - 1], ['Head 1', f'Head {n_heads}'])
ax.set_yticks([0, 0.1, 0.2])
ax.set_ylim((-0.05, 0.3))

ax.spines[['right', 'top']].set_visible(False)
for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2)
ax.tick_params(width=2)

ax.legend(frameon=False)
fig.tight_layout()

# plt.show()
filename = f'layer{model_layer}_{file_prefix}amr'
fig.savefig(f'../../results/figs/reading_brain/{model_type}/{model_size}/{filename}.png', dpi=80)
