import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

model_type = 'llama'
model_size = '30B'
part = None
method = 'pearson'
p = 2
residue = True
head_pool = 'mean'

d_size_layer = {
    'gpt2-base': 12,
    'gpt2-large': 36,
    'gpt2-multi': 24,
    'bert-base': 12,
    'bert-large': 24,
    't5-base': 12,
    't5-large': 24,
    'llama-7B': 32,
    'llama-13B': 40,
    'llama-30B': 60
}

file_prefix = 'residue_' if residue else ''

if model_type in ['llama', 'alpaca', 'vicuna']:
    n_layers = d_size_layer['llama-' + model_size]
else:
    n_layers = d_size_layer[model_type + '-' + model_size]

font = {'family': 'DejaVu Sans',
        'weight': 'normal',
        'size': 25}

matplotlib.rc('font', **font)

labels = [f'Layer {i}' for i in range(n_layers)]
r_num_gpt_means = []
r_num_gpt_error = []
r_num_random_means = []
r_num_random_error = []
r_dur_gpt_means = []
r_dur_gpt_error = []
r_dur_random_means = []
r_dur_random_error = []

for layer in range(n_layers):
    df_gpt = pd.read_excel(
        f'../../results/correlation/{model_type}/{model_size}/{head_pool}_{file_prefix}m_{file_prefix}s_sac_p{p}_{method}.xlsx',
        sheet_name=f'layer {layer}'
    )
    df_random = pd.read_excel(
        f'../../results/correlation/{model_type}/{model_size}-random/{head_pool}_{file_prefix}m_{file_prefix}s_sac_p{p}-random_{method}.xlsx',
        sheet_name=f'layer {layer}')

    r_num_gpt = df_gpt['r number'].to_numpy()
    r_num_gpt_means.append(np.mean(r_num_gpt))
    r_num_gpt_error.append(stats.sem(r_num_gpt))

    r_num_random = df_random['r number'].to_numpy()
    r_num_random_means.append(np.mean(r_num_random))
    r_num_random_error.append(stats.sem(r_num_random))

    r_dur_gpt = df_gpt['r duration'].to_numpy()
    r_dur_gpt_means.append(np.mean(r_dur_gpt))
    r_dur_gpt_error.append(stats.sem(r_dur_gpt))

    r_dur_random = df_random['r duration'].to_numpy()
    r_dur_random_means.append(np.mean(r_dur_random))
    r_dur_random_error.append(stats.sem(r_dur_random))


print(f'number, {residue}, {p}')
print(f'Max Corr: {np.max(r_num_gpt_means)} in layer {np.argmax(r_num_gpt_means)} for {model_type} {model_size}')
x = np.arange(len(labels))  # the label locations
width = 0.22  # the width of the bars

fig, ax = plt.subplots(figsize=(14, 7))
rects1 = ax.bar(x - 1.5 * width, r_num_gpt_means, width, yerr=r_num_gpt_error,
                label=f'Saccade times vs. {model_type} ({model_size})', color='xkcd:soft blue', ecolor='tab:gray')
rects2 = ax.bar(x - 0.5 * width, r_dur_gpt_means, width, yerr=r_dur_gpt_error,
                label=f'Fixation duration vs. {model_type} ({model_size})', color='xkcd:reddish', ecolor='tab:gray')
# rects3 = ax.bar(x + 0.5 * width, r_num_random_means, width, yerr=r_num_random_error,
#                 label='Saccade times vs. random', color='xkcd:light peach', ecolor='tab:gray')
# rects4 = ax.bar(x + 1.5 * width, r_dur_random_means, width, yerr=r_dur_random_error,
#                 label='Fixation duration vs. random', color='xkcd:pale olive', ecolor='tab:gray')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel(f"{model_type.upper()}-{model_size.capitalize()} Layers")
ax.set_ylabel(f"Mean {method.capitalize()}'s" + r"$\it{r}$")
ax.set_title(f"Saccade vs. {model_type.upper()} Attention (L{p})")
ax.set_xticks([0, n_layers - 1], ['Layer 1', f'Layer {n_layers}'])
ax.set_yticks([0, 0.1, 0.2])
ax.set_ylim((-0.05, 0.3))

ax.spines[['right', 'top']].set_visible(False)
for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2)
ax.tick_params(width=2)

ax.legend(frameon=False)
fig.tight_layout()

# plt.show()
filename = f'{head_pool}_{file_prefix}{p}'
fig.savefig(f'../../results/figs/reading_brain/{model_type}/{model_size}/{filename}.png', dpi=80)
