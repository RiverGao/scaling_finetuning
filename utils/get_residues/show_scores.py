import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import sys
import matplotlib.ticker as mtick


data_type = 'rb'  # amr or rb
attn_method = ''

d_size_layer = {'7B': 32, '13B': 40}
font = {'family': 'DejaVu Sans',
        'weight': 'normal',
        'size': 25}
matplotlib.rc('font', **font)

scores_list = [[], [], []]  # 3 models * 2 sizes
for size in ['7B', '13B']:
    n_layer = d_size_layer[size]
    x = np.arange(n_layer)
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ['black', 'xkcd:soft blue', 'xkcd:reddish']

    for im, model in enumerate(['LLaMA', 'Alpaca', 'Vicuna']):
        # 3 lines
        suffix = 'amr' if data_type == 'amr' else 'rb_p1'
        y = np.load(f'../../results/lr_scores_reading_brain/{model.lower()}/{size}/{attn_method + "_" if attn_method else ""}scores_{suffix}.npy')
        scores_list[im].append(y)
        line = ax.plot(x, y, label=model, color=colors[im])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Model Layers')
    ax.set_ylabel(f'Regression Scores')
    ax.set_title(f'{data_type.upper()} Trivial Pattern Reliance of {size} Models')

    ax.set_xticks([0, n_layer - 1], ['Layer 0', f'Layer {n_layer - 1}'])
    # ax.set_yticks([0, 0.025, 0.05])
    # ax.set_ylim((0, 0.37))

    ax.spines[['right', 'top']].set_visible(False)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(2)
    ax.tick_params(width=2)

    ax.legend(frameon=False)
    fig.tight_layout()

    # plt.show()
    folder = 'amr' if data_type == 'amr' else 'reading_brain'
    filename = f'scores_{size}'
    fig.savefig(f'../../results/figs/{folder}/lr-scores/{attn_method + "_" if attn_method else ""}{filename}.png', dpi=120)
    plt.close(fig)

# show difference in sizes of 7B and 13B
y1 = []
y2 = []
y3 = []
for part in range(4):
    llama7B_scores_in_part = []
    alpaca7B_scores_in_part = []
    vicuna7B_scores_in_part = []
    for layer in range(part * 8, (part + 1) * 8):
        llama7B_scores_in_part.append(scores_list[0][0][layer])
        alpaca7B_scores_in_part.append(scores_list[1][0][layer])
        vicuna7B_scores_in_part.append(scores_list[2][0][layer])

    llama13B_scores_in_part = []
    alpaca13B_scores_in_part = []
    vicuna13B_scores_in_part = []
    for layer in range(part * 10, (part + 1) * 10):
        llama13B_scores_in_part.append(scores_list[0][1][layer])
        alpaca13B_scores_in_part.append(scores_list[1][1][layer])
        vicuna13B_scores_in_part.append(scores_list[2][1][layer])

    y1.append((np.mean(llama13B_scores_in_part) - np.mean(llama7B_scores_in_part)) / np.mean(llama7B_scores_in_part))
    y2.append((np.mean(alpaca13B_scores_in_part) - np.mean(alpaca7B_scores_in_part)) / np.mean(alpaca7B_scores_in_part))
    y3.append((np.mean(vicuna13B_scores_in_part) - np.mean(vicuna7B_scores_in_part)) / np.mean(vicuna7B_scores_in_part))


x = np.arange(4)
width = 0.1
fig, ax = plt.subplots(figsize=(14, 7))
colors = ['black', 'xkcd:soft blue', 'xkcd:reddish']
hatches = ['/', '/', '.']
for iy, (yi, mi) in enumerate(zip([y1, y2, y3], ['LLaMA', 'Alpaca', 'Vicuna'])):
    # line = ax.plot(x, yi, label=mi, color=colors[iy])
    rects = ax.bar(
        x - (1 - iy) * width, yi,
        width, label=mi,
        color=colors[iy], edgecolor='w',
        # fill=True, hatch=hatches[iy]
    )

line = ax.axhline(0, 0, x[-1], linewidth=0.5, color='tab:gray')

ax.set_xlabel('Layer Quarters')
ax.set_ylabel(f'Relative Difference')
ax.set_title(r'$\Delta$ Trivial Pattern Reliance (13B - 7B)')

ax.set_xticks([0, 1, 2, 3], ['25%', '50%', '75%', '100%'])
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
ax.spines[['right', 'top']].set_visible(False)
for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2)
ax.tick_params(width=2)
# ax.set_ylim((-0.05, 0.05))

ax.legend(frameon=False)
fig.tight_layout()

# plt.show()
folder = 'amr' if data_type == 'amr' else 'reading_brain'
filename = f'scores_{size}'
fig.savefig(f'../../results/figs/{folder}/lr-scores/{attn_method + "_" if attn_method else ""}trivial-size.png', dpi=120)
plt.close(fig)


# show difference in sizes of tuned and untuned models
y1 = []
y2 = []
y3 = []
y4 = []
for part in range(4):
    llama7B_scores_in_part = []
    alpaca7B_scores_in_part = []
    vicuna7B_scores_in_part = []
    for layer in range(part * 8, (part + 1) * 8):
        llama7B_scores_in_part.append(scores_list[0][0][layer])
        alpaca7B_scores_in_part.append(scores_list[1][0][layer])
        vicuna7B_scores_in_part.append(scores_list[2][0][layer])

    llama13B_scores_in_part = []
    alpaca13B_scores_in_part = []
    vicuna13B_scores_in_part = []
    for layer in range(part * 10, (part + 1) * 10):
        llama13B_scores_in_part.append(scores_list[0][1][layer])
        alpaca13B_scores_in_part.append(scores_list[1][1][layer])
        vicuna13B_scores_in_part.append(scores_list[2][1][layer])

    baseline7B = np.mean(llama7B_scores_in_part)
    baseline13B = np.mean(llama13B_scores_in_part)
    y1.append((np.mean(alpaca7B_scores_in_part) - baseline7B) / baseline7B)
    y2.append((np.mean(vicuna7B_scores_in_part) - baseline7B) / baseline7B)
    y3.append((np.mean(alpaca13B_scores_in_part) - baseline13B) / baseline13B)
    y4.append((np.mean(vicuna13B_scores_in_part) - baseline13B) / baseline13B)


ys = [[y1, y2], [y3, y4]]
colors = [['xkcd:pale olive', 'xkcd:light peach'], ['xkcd:soft blue', 'xkcd:reddish']]
hatches = [['/', '.'], ['//', '..']]
fig, ax = plt.subplots(figsize=(14, 7))
for i, size in enumerate(['7B', '13B']):
    x = np.arange(4)
    width = 0.1
    for iy, (yi, mi) in enumerate(zip(ys[i], [f'Alpaca-LLaMA({size})', f'Vicuna-LLaMA({size})'])):
        # line = ax.plot(x, yi, label=mi, color=colors[iy])
        offset = 1 if i == 0 else -1
        rects = ax.bar(
            x - (offset + 1 - iy) * width, yi,
            width, label=mi,
            color=colors[i][iy], edgecolor='w',
            # fill=True, hatch=hatches[i][iy]
        )

line = ax.axhline(0, 0, x[-1], linewidth=0.5, color='tab:gray')

ax.set_xlabel('Layer Quarters')
ax.set_ylabel(f'Relative Difference')
ax.set_title(r'$\Delta$ Trivial Pattern Reliance (Tuned - Pretrained)')

ax.set_xticks([0, 1, 2, 3], ['25%', '50%', '75%', '100%'])
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))

ax.spines[['right', 'top']].set_visible(False)
for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2)
ax.tick_params(width=2)
# ax.set_ylim((-0.02, 0.02))

ax.legend(frameon=False)
fig.tight_layout()

# plt.show()
folder = 'amr' if data_type == 'amr' else 'reading_brain'
filename = f'scores_{size}'
fig.savefig(f'../../results/figs/{folder}/lr-scores/{attn_method + "_" if attn_method else ""}trivial-name.png', dpi=120)
plt.close(fig)
