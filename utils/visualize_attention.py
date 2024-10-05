import numpy as np
import pickle
import matplotlib
from matplotlib import pyplot as plt


def symmetry(mat):
    # get the symmetric version of under triangle matrices
    diag = np.diag(np.diag(mat))
    upper = mat.T
    result = mat + upper - diag
    return result


model_type = 'llama'
model_size = '7B'
part = None
layer = 9

d_size_layer = {
    'gpt2-base': 12,
    'gpt2-large': 36,
    'gpt2-multi': 24,
    'bert-base': 12,
    'bert-large': 24,
    't5-base': 12,
    't5-large': 24,
    'llama-7B': 32,
    'llama-13B': 40
}
if model_type in ['llama', 'alpaca', 'vicuna']:
    n_layers = d_size_layer['llama-' + model_size]
else:
    n_layers = d_size_layer[model_type + '-' + model_size]

font = {'family': 'DejaVu Sans',
        'weight': 'normal',
        'size': 18}
matplotlib.rc('font', **font)

article_id = 0
sentence_id = 16

with open(f'../golden_attention/amr/amr_matrices.pkl', 'rb') as f:
    amr_connections = pickle.load(f)  # sentence, word, word
sent_connections = amr_connections[sentence_id]

with open('../sentence_data/amr3.0/sample_data/sample-sent.txt', 'r') as f:
    sentences = f.read().split('\n')
sent_words = sentences[sentence_id].strip().split()
n_words = len(sent_words)

print(sent_words)

fig, axs = plt.subplots(1, 2, figsize=(12, 9))
amr_mat = axs[0].matshow(sent_connections)
axs[0].set_xticks([i for i in range(len(sent_words))], [w for w in sent_words], rotation=45, ha='left')
axs[0].set_yticks([i for i in range(len(sent_words))], [w for w in sent_words], rotation=45, ha='right')

# (n_sents, max_sent_len, max_sent_len)
model_layer_attn = np.load(
    f'../model_attention/amr/{model_type}/{model_size}/amr_layer{layer}.npy')

attn_mean = model_layer_attn[sentence_id].mean(axis=0)[: n_words, : n_words]
attn_mean = symmetry(attn_mean)
attn_mean_mat = axs[1].matshow(attn_mean)
axs[1].set_xticks([i for i in range(len(sent_words))], [w for w in sent_words], rotation=45, ha='left')
axs[1].set_yticks([i for i in range(len(sent_words))], [w for w in sent_words], rotation=45, ha='right')

# for head in range(32):
#     model_head_attn = model_layer_attn[sentence_id, head, : n_words, :n_words]
#     model_head_attn = symmetry(model_head_attn)
#     # threshold = 1 / model_layer_attn.shape[0]
#     # threshold_attn = (model_layer_attn > threshold) * model_layer_attn
#     i = (head + 2) // 9
#     j = (head + 2) % 9
#     attn_mat = axs[i, j].matshow(model_head_attn)
#     # attn_mat = ax2.matshow(threshold_attn)
#     # axs[0, 1].set_xticks([i for i in range(len(sent_words))], [w for w in sent_words], rotation=45, ha='left')
#     # axs[0, 1].set_yticks([i for i in range(len(sent_words))], [w for w in sent_words], rotation=45, ha='right')
# # ax2.tick_params(labelrotation=45)

# fig.colorbar(attn_mat, ax=ax2)
fig.tight_layout()
plt.show()

# fig.savefig(f'/Users/river/Desktop/fig1.png', dpi=150, transparent=True)
plt.close(fig)

# for layer in n_layers:
#     model_layer_attn = np.load(
#         f'attention/model2/{model_size}/attention_amr_layer{layer}.npy')  # (n_sents, max_sent_len, max_sent_len)


