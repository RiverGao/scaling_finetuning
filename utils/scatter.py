import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

font = {'family': 'DejaVu Sans',
        'weight': 'normal',
        'size': 20}
matplotlib.rc('font', **font)

# saccade
# ntp loss: gpt2, llama7b, llama13b, llama30b, llama65b, alpaca7b, alpaca13b, vicuna7b, vicuna13b
x = np.array([0.3264, 0.2408, 0.2406, 0.2372, 0.2375, 0.2646, 0.2634, 0.2593, 0.2847])
y1 = np.array([34.99, 53.04, 55.66, 63.16, 64.05, 52.51, 55.05, 51.9, 54.26])
y2 = np.array([40.03, 62.44, 64.2, 69.4, 70.07, 61.71, 63.46, 61.19, 61.31])
# log scale
# x = np.log(np.array([0.774, 7, 13, 33, 65]))
# y1 = np.array([34.99, 53.04, 55.66, 63.16, 64.05])
# y2 = np.array([40.03, 62.44, 64.2, 69.4, 70.07])

# fmri
# Max score / ceiling of gpt2_large: 0.05894234178524434
# Max score / ceiling of llama_7B: 0.15445947287999293
# Max score / ceiling of llama_13B: 0.20194667370429056
# Max score / ceiling of llama_30B: 0.22225170504933656
# Max score / ceiling of llama_65B: 0.2716109174278264
# ntp loss
x = np.array([0.3264, 0.2408, 0.2406, 0.2372, 0.2375, 0.2646, 0.2634, 0.2593, 0.2847])
y1 = np.array([4.64, 11.02, 14.69, 16.78, 18.59, 10.99, 14.67, 11.35, 14.02])
y2 = np.array([5.89, 15.45, 20.19, 22.23, 27.16, 15.61, 19.44, 15.44, 18.57])
# log scale
# x = np.log(np.array([0.774, 7, 13, 33, 65]))
# y1 = np.array([4.64, 11.02, 14.69, 16.78, 18.59])
# y2 = np.array([5.89, 15.45, 20.19, 22.23, 27.16])

# Initialize layout
fig, ax = plt.subplots(figsize=(11, 6))

# Add scatterplot
ax.scatter(x, y1, s=12**2, marker='P', color='xkcd:soft blue', label='L1')
ax.scatter(x, y2, s=12**2,  marker='X', color='xkcd:reddish', label='L2')

# Fit linear regression via least squares with numpy.polyfit
# It returns an slope (b) and intercept (a)
# deg=1 means linear fit (i.e. polynomial of degree 1)
w1, b1 = np.polyfit(x, y1, deg=1)
w2, b2 = np.polyfit(x, y2, deg=1)

# Create sequence of 100 numbers from 0 to 100
xseq = np.linspace(np.min(x), np.max(x), num=100)

# Plot regression line
ax.plot(xseq, w1 * xseq + b1, '--', color='xkcd:soft blue', linewidth=3)
ax.plot(xseq, w2 * xseq + b2, '--', color='xkcd:reddish', linewidth=3)

ax.set_xlabel('Per-Token Loss')
ax.set_ylabel('Max Human Resemblance')
ax.set_title('Correlation between NTP Loss and Human Resemblance')
# ax.set_xlabel('Log Parameter Scale (B)')
# ax.set_ylabel('Max Human Resemblance')
# ax.set_title('Correlation between Log Scale and Human Resemblance')

ax.spines[['right', 'top']].set_visible(False)
for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2)
ax.tick_params(width=2)
ax.legend(frameon=False)
fig.tight_layout()

# plt.show()
fig.savefig(f'../results/figs/reading_brain/interpret/ntp_vs_score_fmri', dpi=120)
# fig.savefig(f'../results/figs/reading_brain/interpret/scale_vs_score', dpi=120)
plt.close(fig)


r1, p1 = pearsonr(x, y1)
r2, p2 = pearsonr(x, y2)
print(f'L1:\tr={r1}\tp={p1}\nL2:\tr={r2}\tp={p2}')

