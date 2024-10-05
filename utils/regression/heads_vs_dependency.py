import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
import sys


def tril_idx(n):
    # 00, 01, 11, 02, 12, 22
    x = []
    y = []
    for i in range(n):
        for j in range(n):
            if j > i:
                break
            x.append(i)
            y.append(j)
    return np.array(x), np.array(y)


model_name = 'gpt2'
model_size = 'large'
residue = False

d_size_layers = {'large': 36, '7B': 32, '13B': 40, '30B': 60}
d_size_heads = {'large': 20, '7B': 32, '13B': 40, '30B': 52}
np.random.seed(42)
file_prefix = 'residue_' if residue else ''
n_layers = d_size_layers[model_size]
n_heads = d_size_heads[model_size]

with open('../../golden_attention/amr/dependency_matrices.pkl', 'rb') as f:
    dep_data = pickle.load(f)  # n_sentence, word, word

sentence_lengths = []  # store the length of each sentence
r2_scores = []  # R^2 of LR for each layer
n_sentences = len(dep_data)
flatten_dep_data = []  # n_sentences * n_words * (n_words + 1) / 2

for layer in range(n_layers):
    model_layer_attn = np.load(
        f'../../model_attention/amr/{model_name}/{model_size}/{file_prefix}amr_layer{layer}.npy'
    )  # (max_n_sents, n_head, max_sent_len, max_sent_len)
    assert not np.isnan(np.sum(model_layer_attn)), f'Layer {layer} has NaN values'
    flatten_model_attn = [[] for i in range(n_heads)]  # length: 2-D, n_sentences * n_words * (n_words + 1) / 2, n_heads

    for sentj in range(n_sentences):
        # flatten dep data, only do it once
        if layer == 0:
            sentence_data = np.array(dep_data[sentj])  # (word, word)
            n_words = len(sentence_data)
            sentence_lengths.append(n_words)

            tril_x, tril_y = tril_idx(n_words)
            flat_sentence_data = sentence_data[tril_x, tril_y].tolist()  # n_words * (n_words + 1) / 2
            flatten_dep_data.extend(flat_sentence_data)

        # flattened indices
        n_words = sentence_lengths[sentj]
        tril_x, tril_y = tril_idx(n_words)
        # flatten attentions in each head
        for head in range(n_heads):
            sentence_head_attn = model_layer_attn[sentj, head, :, :]  # (max_sent_len, max_sent_len)
            flat_sentence_head_attn = sentence_head_attn[tril_x, tril_y].tolist()  # n_words * (n_words + 1) / 2
            flatten_model_attn[head].extend(flat_sentence_head_attn)

    # do regression for this layer
    X = np.array(flatten_model_attn).T
    y = np.array(flatten_dep_data)
    reg_layer = LinearRegression().fit(X, y)
    score = reg_layer.score(X, y)
    r2_scores.append(score)

np.save(f'../../results/lr_scores/heads_vs_dependency/{file_prefix}{model_name}_{model_size}.npy', np.array(r2_scores))
print(np.mean(r2_scores))













