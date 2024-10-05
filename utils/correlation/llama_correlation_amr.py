import numpy as np
from scipy import stats
import pickle
import warnings

warnings.simplefilter(action='ignore')
import pandas as pd
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


# model_name = sys.argv[1]
# model_size = sys.argv[2]
# residue = eval(sys.argv[3])
# head_pool = sys.argv[4]

model_name = 'llama'
model_size = '13B'
residue = False
head_pool = 'mean'

d_size_layer = {'7B': 32, '13B': 40, '30B': 60, 'large': 36}
n_layers = d_size_layer[model_size]
method = 'pearson'

np.random.seed(42)
file_prefix = 'residue_' if residue else ''

with open(f'../../golden_attention/amr/{file_prefix}amr_matrices.pkl', 'rb') as f:
    amr_connections = pickle.load(f)  # sentence, word, word

with pd.ExcelWriter(
        f'../../results/correlation/{model_name}/{model_size}/{head_pool}_{file_prefix}amr_{method}.xlsx') as writer_llama:
    with pd.ExcelWriter(
            f'../../results/correlation/{model_name}/{model_size}-random/{head_pool}_{file_prefix}amr_{method}.xlsx') as writer_random:
        for layer in range(n_layers):
            df_llama = pd.DataFrame(columns=['r AMR'])
            df_random = pd.DataFrame(columns=['r AMR'])
            sheet_name = f'layer {layer}'
            llama_layer_attn = np.load(
                f'../../model_attention/amr/{model_name}/{model_size}/{file_prefix}amr_layer{layer}.npy')
            # (n_sents, n_head, max_sent_len, max_sent_len)
            assert not np.isnan(np.sum(llama_layer_attn)), layer

            for senti in range(len(amr_connections)):
                if amr_connections[senti].shape[0] < 2:
                    continue
                corrs_amr_llama = []
                corrs_amr_random = []

                r_amr = []
                r_amr_random = []

                mati = amr_connections[senti]
                llama_layer_senti = llama_layer_attn[senti]

                if head_pool == 'mean':
                    llama_layer_senti = llama_layer_senti.mean(axis=0)
                elif head_pool == 'max':
                    llama_layer_senti = llama_layer_senti.max(axis=0)
                else:
                    raise ValueError(f'Unknown head pooling: {head_pool}')

                n_word = mati.shape[0]
                idx_tril = tril_idx(n_word)
                # llama_layer_arti_sentj = llama_layer_arti_sentj[1:, 1:]

                vec_llama = llama_layer_senti[idx_tril]
                vec_random = vec_llama.copy()
                np.random.shuffle(vec_random)

                vec_amr = mati[idx_tril]

                # print(senti)
                corr_func = stats.pearsonr if method == 'pearson' else stats.spearmanr

                r_senti_amr, p_senti_amr = corr_func(vec_amr, vec_llama)
                if r_senti_amr == r_senti_amr:
                    r_amr.append(r_senti_amr)
                    df_llama = df_llama.append(
                        {'r AMR': r_senti_amr, 'p AMR': p_senti_amr},
                        ignore_index=True)

                r_senti_amr_random, p_senti_amr_random = corr_func(vec_amr, vec_random)
                if r_senti_amr_random == r_senti_amr_random:
                    r_amr_random.append(r_senti_amr_random)
                    df_random = df_random.append(
                        {'r AMR': r_senti_amr_random, 'p AMR': p_senti_amr_random},
                        ignore_index=True)

                # r_amr_mean = np.mean(r_amr)
                # r_amr_random_mean = np.mean(r_amr_random)
                # df_llama = df_llama.append(
                #     {'r AMR': r_amr_mean, 'p AMR': p_senti_amr},
                #     ignore_index=True)
                # df_random = df_random.append(
                #     {'r AMR': r_amr_random_mean, 'p AMR': p_senti_amr_random},
                #     ignore_index=True)

            df_llama.to_excel(writer_llama, sheet_name=sheet_name, index=False)
            df_random.to_excel(writer_random, sheet_name=sheet_name, index=False)
