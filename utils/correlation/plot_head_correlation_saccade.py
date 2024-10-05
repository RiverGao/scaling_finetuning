import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

model_type = 'llama'
model_size = '30B'
model_layer = 45
method = 'pearson'
p = 2
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
r_num_gpt_means = []
r_num_gpt_error = []
r_dur_gpt_means = []
r_dur_gpt_error = []

for head in range(n_heads):
    df_gpt = pd.read_excel(
        f'../../results/correlation/{model_type}/{model_size}/layer{model_layer}_{file_prefix}sac_p{p}_{method}.xlsx',
        sheet_name=f'head {head}'
    )

    r_num_gpt = df_gpt['r number'].to_numpy()
    r_num_gpt_means.append(np.mean(r_num_gpt))
    r_num_gpt_error.append(stats.sem(r_num_gpt))

    r_dur_gpt = df_gpt['r duration'].to_numpy()
    r_dur_gpt_means.append(np.mean(r_dur_gpt))
    r_dur_gpt_error.append(stats.sem(r_dur_gpt))


print(f'number, {residue}, {p}')
print(f'Max Corr: {np.max(r_num_gpt_means)} in head {np.argmax(r_num_gpt_means)} for {model_type} {model_size} layer {model_layer}')
x = np.arange(len(labels))  # the label locations
width = 0.22  # the width of the bars

fig, ax = plt.subplots(figsize=(14, 7))
rects1 = ax.bar(x - 1.5 * width, r_num_gpt_means, width, yerr=r_num_gpt_error,
                label=f'Saccade times vs. {model_type} ({model_size})', color='xkcd:soft blue', ecolor='tab:gray')
rects2 = ax.bar(x - 0.5 * width, r_dur_gpt_means, width, yerr=r_dur_gpt_error,
                label=f'Fixation duration vs. {model_type} ({model_size})', color='xkcd:reddish', ecolor='tab:gray')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel(f"{model_type.upper()}-{model_size.capitalize()} Layers")
ax.set_ylabel(f"Mean {method.capitalize()}'s" + r"$\it{r}$")
ax.set_title(f"Saccade vs. {model_type.upper()} Attention (L{p})")
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
filename = f'layer{model_layer}_{file_prefix}sac_{p}'
fig.savefig(f'../../results/figs/reading_brain/{model_type}/{model_size}/{filename}.png', dpi=80)
