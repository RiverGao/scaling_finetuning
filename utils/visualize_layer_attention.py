import numpy as np
import pickle
from matplotlib import pyplot as plt


def symmetry(mat):
    # get the symmetric version of under triangle matrices
    diag = np.diag(np.diag(mat))
    upper = mat.T
    result = mat + upper - diag
    return result


model_type = 'llama'
model_size = '7B'

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

article_id = 1
sentence_id = 15
layer = 0
n_head = n_layers

# with open(f'../golden_attention/amr/amr_matrices.pkl', 'rb') as f:
#     amr_connections = pickle.load(f)  # sentence, word, word
# sent_connections = amr_connections[sentence_id]

with open('../sentence_data/reading_brain/sentences_p1.txt', 'r') as f:
    articles = f.read().split('\n\n')
sentences = [a.strip().split('\n') for a in articles]

sent_words = sentences[article_id][sentence_id].strip().split()
n_words = len(sent_words)

print(sent_words)

fig, axs = plt.subplots(5, 8, figsize=(25, 18))
# amr_mat = axs[0, 0].matshow(sent_connections)
# axs[0, 0].set_xticks([i for i in range(len(sent_words))], [w for w in sent_words], rotation=45, ha='left')
# axs[0, 0].set_yticks([i for i in range(len(sent_words))], [w for w in sent_words], rotation=45, ha='right')
# axs[0, 0].set_title(f'{model_type} {model_size} Layer {layer} Attention on RB A{article_id}s{sentence_id}')

for layer in range(n_layers):
    # model_layer_attn = np.load(
    #     f'../model_attention/reading_brain/llama/13B/p1/rb_p1_layer{layer}.npy'
    # )  # (5, n_sents, n_head, max_sent_len, max_sent_len)

    model_layer_attn = np.load(
        f'../model_attention/reading_brain/{model_type}/{model_size}/flow/rb_p1_layer{layer}.npy'
    )  # (5, n_sents, 1, max_sent_len, max_sent_len)

    attn_mean = model_layer_attn[article_id, sentence_id, 0, : n_words, : n_words]
    # attn_mean = symmetry(attn_mean)

    # i = (layer + 1) // 5
    # j = (layer + 1) % 5
    i = layer // 8
    j = layer % 8
    print(i, j)
    attn_mean_mat = axs[i, j].matshow(attn_mean)
    axs[i, j].set_xticks([i for i in range(len(sent_words))], [w for w in sent_words], rotation=45, ha='left')
    axs[i, j].set_yticks([i for i in range(len(sent_words))], [w for w in sent_words], rotation=45, ha='right')
    axs[i, j].set_xlabel('Word To')
    axs[i, j].set_ylabel('Word From')
    axs[i, j].set_title(f'Layer {layer}')

# model_layer_attn = np.load(
#     f'../model_attention/reading_brain/{model_type}/{model_size}/p1/rb_p1_layer{layer}.npy'
# )  # (n_sents, n_head, max_sent_len, max_sent_len)
# attn = model_layer_attn[article_id, sentence_id, :,  :n_words, : n_words]
#
# for head in range(n_head):
#     # attn_mean = symmetry(attn_mean)
#     i = head // 8
#     j = head % 8
#     attn_mat = axs[i, j].matshow(attn[head, :, :])
#     axs[i, j].set_xticks([i for i in range(len(sent_words))], [w for w in sent_words], rotation=45, ha='left')
#     axs[i, j].set_yticks([i for i in range(len(sent_words))], [w for w in sent_words], rotation=45, ha='right')
#     axs[i, j].set_xlabel('Word To')
#     axs[i, j].set_ylabel('Word From')
#     axs[i, j].set_title(f'Head {head}')


fig.tight_layout()
plt.show()
# fig.savefig(f'../results/figs/reading_brain/{model_type}/{model_size}/a{article_id}s{sentence_id}l{layer}.png', dpi=80)
# plt.close(fig)

