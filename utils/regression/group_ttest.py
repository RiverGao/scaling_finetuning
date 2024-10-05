import numpy as np
from scipy import stats

view = 'num'  # dur or num
model_name = 'vicuna'
model_size = '7B'

d_size_layers = {'large': 36, '7B': 32, '13B': 40, '30B': 60}
np.random.seed(42)
n_layers = d_size_layers[model_size]

l1_scores = np.load(f'../../results/lr_scores/heads_vs_saccade/num_1_{model_name}_{model_size}.npy')
l2_scores = np.load(f'../../results/lr_scores/heads_vs_saccade/num_2_{model_name}_{model_size}.npy')

count_ttest = 0
t_values = []
p_values = []
for layer in range(n_layers):
    layer_l1_scores = np.array(l1_scores[layer])
    layer_l2_scores = np.array(l2_scores[layer])

    t_layer, p_layer = stats.ttest_ind(layer_l1_scores, layer_l2_scores, alternative='less')
    t_values.append(t_layer)
    p_values.append(p_layer)
    count_ttest += 1

significant_p = 0.05 / count_ttest
for i, (t, p) in enumerate(zip(t_values, p_values)):
    if p >= significant_p:
        print(f'Layer {i} is not significant with t={t:.05}, p={p:.05}')



