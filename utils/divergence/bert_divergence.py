import numpy as np
from scipy.stats import entropy as kl_div
from scipy.special import softmax
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd


# def to_probability(m: np.array):
#     # problem: cannot handle all-zero rows
#     assert len(m.shape) == 2
#     assert m.shape[0] == m.shape[1]
#     n = m.shape[0]
#     row_sums = np.sum(m, axis=1)
#     e = 1 / row_sums * np.eye(n)
#     m_norm = np.dot(e, m)
#     return m_norm


model_size = 'large'
d_size_layer = {'base': 12, 'large': 24}
n_layers = d_size_layer[model_size]
method = 'KL'
np.random.seed(42)

with open(f'amr_matrices.pkl', 'rb') as f:
    amr_connections = pickle.load(f)  # sentence, word, word

with pd.ExcelWriter(f'divergence/bert/{model_size}/amr_{model_size}_{method}.xlsx') as writer_bert:
    with pd.ExcelWriter(f'divergence/bert/{model_size}-random/amr_{model_size}-random_{method}.xlsx') as writer_random:
        for layer in range(n_layers):
            df_bert = pd.DataFrame(columns=['divergence'])
            df_random = pd.DataFrame(columns=['divergence'])
            sheet_name = f'layer {layer}'
            bert_layer_attn = np.load(
                f'attention/bert/{model_size}/attention_amr_layer{layer}.npy')  # (n_sents, max_sent_len, max_sent_len)

            for senti in range(len(amr_connections)):
                if amr_connections[senti].shape[0] < 2:
                    continue

                mati = softmax(amr_connections[senti], axis=1)
                # mati = to_probability(amr_connections[senti])
                bert_layer_senti = bert_layer_attn[senti]

                n_word = mati.shape[0]
                # idx_tril = tril_idx(n_word)

                if method == 'KL':
                    vec_bert = bert_layer_senti[:n_word, :n_word].flatten()

                    # vec_random1 = vec_bert.copy()
                    # vec_random2 = vec_bert.copy()
                    # vec_random3 = vec_bert.copy()
                    # np.random.shuffle(vec_random1)
                    # np.random.shuffle(vec_random2)
                    # np.random.shuffle(vec_random3)
                    # vec_random = np.mean([vec_random1, vec_random2, vec_random3], axis=0)

                    vec_random = vec_bert.copy()
                    np.random.shuffle(vec_random)

                    # vec_random = softmax(np.zeros(shape=(n_word, n_word))).flatten()  # completely uniform

                    vec_amr = mati.flatten()

                    # print(senti)
                    divergence = kl_div(vec_amr, vec_bert)
                    divergence_random = kl_div(vec_amr, vec_random)

                elif method == 'norm':
                    mat_bert = bert_layer_senti[:n_word, :n_word]

                    # mat_random1 = mat_bert.copy()
                    # mat_random2 = mat_bert.copy()
                    # mat_random3 = mat_bert.copy()
                    # np.random.shuffle(mat_random1)
                    # np.random.shuffle(mat_random2)
                    # np.random.shuffle(mat_random3)
                    # mat_random = np.mean([mat_random1, mat_random2, mat_random3], axis=0)

                    mat_random_t = mat_bert.copy().T
                    np.random.shuffle(mat_random_t)
                    mat_random = mat_random_t.T

                    divergence = np.linalg.norm(mat_bert - mati, 'fro')
                    divergence_random = np.linalg.norm(mat_random - mati, 'fro')

                if divergence < 1000 and divergence_random < 1000:
                    df_bert = df_bert.append(
                        {'divergence': divergence},
                        ignore_index=True)
                    df_random = df_random.append(
                        {'divergence': divergence_random},
                        ignore_index=True)

            df_bert.to_excel(writer_bert, sheet_name=sheet_name, index=False)
            df_random.to_excel(writer_random, sheet_name=sheet_name, index=False)
