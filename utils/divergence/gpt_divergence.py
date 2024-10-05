import numpy as np
from scipy.stats import entropy as kl_div
from scipy.special import softmax
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd


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


model_size = 'large'
d_size_layer = {'base': 12, 'large': 36, 'xl': 48, 'multi': 24}
n_layers = d_size_layer[model_size]
method = 'KL'
np.random.seed(42)

with open(f'amr_matrices.pkl', 'rb') as f:
    amr_connections = pickle.load(f)  # sentence, word, word

with pd.ExcelWriter(f'divergence/gpt2/{model_size}/amr_{model_size}_{method}.xlsx') as writer_gpt:
    with pd.ExcelWriter(f'divergence/gpt2/{model_size}-random/amr_{model_size}-random_{method}.xlsx') as writer_random:
        for layer in range(n_layers):
            df_gpt = pd.DataFrame(columns=['divergence'])
            df_random = pd.DataFrame(columns=['divergence'])
            sheet_name = f'layer {layer}'
            gpt_layer_attn = np.load(
                f'attention/gpt2/{model_size}/attention_amr_layer{layer}.npy')  # (n_sents, max_sent_len, max_sent_len)

            for senti in range(len(amr_connections)):
                if amr_connections[senti].shape[0] < 2:
                    continue

                mati = softmax(amr_connections[senti], axis=1)
                gpt_layer_senti = gpt_layer_attn[senti]

                n_word = mati.shape[0]
                idx_tril = tril_idx(n_word)
                # gpt_layer_arti_sentj = gpt_layer_arti_sentj[1:, 1:]

                if method == 'KL':
                    vec_gpt = gpt_layer_senti[idx_tril]
                    vec_random = vec_gpt.copy()
                    np.random.shuffle(vec_random)

                    vec_amr = mati[idx_tril]

                    # print(senti)
                    divergence = kl_div(vec_amr, vec_gpt)
                    divergence_random = kl_div(vec_amr, vec_random)

                elif method == 'norm':
                    mat_gpt = gpt_layer_senti[:n_word, :n_word]
                    mat_amr = np.tril(mati)

                    mat_random_t = mat_gpt.copy().T
                    np.random.shuffle(mat_random_t)
                    mat_random = mat_random_t.T

                    divergence = np.linalg.norm(mat_gpt - mati, 'fro')
                    divergence_random = np.linalg.norm(mat_random - mati, 'fro')

                if divergence < 1000 and divergence_random < 1000:
                    df_gpt = df_gpt.append(
                        {'divergence': divergence},
                        ignore_index=True)
                    df_random = df_random.append(
                        {'divergence': divergence_random},
                        ignore_index=True)

            df_gpt.to_excel(writer_gpt, sheet_name=sheet_name, index=False)
            df_random.to_excel(writer_random, sheet_name=sheet_name, index=False)
