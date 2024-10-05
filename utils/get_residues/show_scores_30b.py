import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import sys


data_type = 'rb'  # amr or rb

font = {'family': 'DejaVu Sans',
        'weight': 'normal',
        'size': 25}
matplotlib.rc('font', **font)

size = '30B'

n_layer = 60
x = np.arange(n_layer)
fig, ax = plt.subplots(figsize=(14, 7))
color = 'black'

y = np.load('../../results/lr_scores_reading_brain/llama/30B/scores_rb_p1.npy')
line = ax.plot(x, y, label='LLaMA', color=color)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Model Layers')
ax.set_ylabel(f'Regression Scores')
ax.set_title(f'{data_type.upper()} Trivial Pattern Predictability of {size} Models')

ax.set_xticks([0, n_layer - 1], ['Layer 0', f'Layer {n_layer - 1}'])
# ax.set_yticks([0, 0.025, 0.05])
ax.set_ylim((0, 0.37))

ax.spines[['right', 'top']].set_visible(False)
for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2)
ax.tick_params(width=2)

ax.legend(frameon=False)
fig.tight_layout()

# plt.show()
folder = 'reading_brain'
filename = f'scores_{size}'
fig.savefig(f'../../results/figs/{folder}/lr-scores/{filename}.png', dpi=80)
plt.close(fig)

# show difference in sizes of 7B and 13B
y1 = []
for part in range(4):
    llama13B_scores_in_part = []
    for layer in range(part * 10, (part + 1) * 10):
        llama13B_scores_in_part.append(y[layer])

    llama30B_scores_in_part = []
    for layer in range(part * 15, (part + 1) * 15):
        llama30B_scores_in_part.append(y[layer])

    y1.append(np.mean(llama30B_scores_in_part) - np.mean(llama13B_scores_in_part))

x = np.arange(4)
width = 0.2
fig, ax = plt.subplots(figsize=(14, 7))
colors = ['black', 'xkcd:soft blue', 'xkcd:reddish']
rects = ax.bar(x, y1, width, label='LLaMA', color=color)

ax.set_xlabel('Layer Quartiles')
ax.set_ylabel(f'Regression Score Differences')
ax.set_title(f'{data_type.upper()} Trivial Pattern Predictability Differences (30B-13B)')

ax.set_xticks([0, 1, 2, 3], ['25%', '50%', '75%', '100%'])
ax.spines[['right', 'top']].set_visible(False)
for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2)
ax.tick_params(width=2)
ax.set_ylim((-0.05, 0.05))

ax.legend(frameon=False)
fig.tight_layout()

# plt.show()
folder = 'reading_brain'
filename = f'scores_{size}'
fig.savefig(f'../../results/figs/{folder}/lr-scores/scores_size_30b.png', dpi=80)
plt.close(fig)

