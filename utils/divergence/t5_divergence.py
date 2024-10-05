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


model_size = 'base'
part = 'decoder'
method = 'KL'

d_size_layer = {'base': 12, 'large': 24}
n_layers = d_size_layer[model_size]
np.random.seed(42)

with open(f'amr_matrices.pkl', 'rb') as f:
    amr_connections = pickle.load(f)  # sentence, word, word

with pd.ExcelWriter(f'divergence/t5/{model_size}/{part}/amr_{model_size}_{method}.xlsx') as writer_t5:
    with pd.ExcelWriter(f'divergence/t5/{model_size}-random/{part}/amr_{model_size}-random_{method}.xlsx') as writer_random:
        for layer in range(n_layers):
            df_t5 = pd.DataFrame(columns=['divergence'])
            df_random = pd.DataFrame(columns=['divergence'])
            sheet_name = f'layer {layer}'
            t5_layer_attn = np.load(
                f'attention/t5/{model_size}/{part}/attention_amr_layer{layer}.npy')  # (n_sents, max_sent_len, max_sent_len)

            for senti in range(len(amr_connections)):
                if amr_connections[senti].shape[0] < 2:
                    continue

                mati = softmax(amr_connections[senti], axis=1)
                # mati = to_probability(amr_connections[senti])
                t5_layer_senti = t5_layer_attn[senti]

                n_word = mati.shape[0]
                # idx_tril = tril_idx(n_word)

                if method == 'KL':
                    vec_t5 = t5_layer_senti[:n_word, :n_word].flatten()

                    # vec_random1 = vec_t5.copy()
                    # vec_random2 = vec_t5.copy()
                    # vec_random3 = vec_t5.copy()
                    # np.random.shuffle(vec_random1)
                    # np.random.shuffle(vec_random2)
                    # np.random.shuffle(vec_random3)
                    # vec_random = np.mean([vec_random1, vec_random2, vec_random3], axis=0)

                    vec_random = vec_t5.copy()
                    np.random.shuffle(vec_random)

                    # vec_random = softmax(np.zeros(shape=(n_word, n_word))).flatten()  # completely uniform

                    vec_amr = mati.flatten()

                    # print(senti)
                    divergence = kl_div(vec_amr, vec_t5)
                    divergence_random = kl_div(vec_amr, vec_random)

                elif method == 'norm':
                    mat_t5 = t5_layer_senti[:n_word, :n_word]

                    # mat_random1 = mat_t5.copy()
                    # mat_random2 = mat_t5.copy()
                    # mat_random3 = mat_t5.copy()
                    # np.random.shuffle(mat_random1)
                    # np.random.shuffle(mat_random2)
                    # np.random.shuffle(mat_random3)
                    # mat_random = np.mean([mat_random1, mat_random2, mat_random3], axis=0)

                    mat_random_t = mat_t5.copy().T
                    np.random.shuffle(mat_random_t)
                    mat_random = mat_random_t.T

                    divergence = np.linalg.norm(mat_t5 - mati, 'fro')
                    divergence_random = np.linalg.norm(mat_random - mati, 'fro')

                if divergence < 1000 and divergence_random < 1000:
                    df_t5 = df_t5.append(
                        {'divergence': divergence},
                        ignore_index=True)
                    df_random = df_random.append(
                        {'divergence': divergence_random},
                        ignore_index=True)

            df_t5.to_excel(writer_t5, sheet_name=sheet_name, index=False)
            df_random.to_excel(writer_random, sheet_name=sheet_name, index=False)
