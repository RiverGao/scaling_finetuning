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

model_name = 'gpt2'
model_size = 'large'
model_layer = 14
residue = True
head_pool = 'mean'

d_size_heads = {'7B': 32, '13B': 40, '30B': 52, 'large': 20}
n_heads = d_size_heads[model_size]
method = 'pearson'

np.random.seed(42)
file_prefix = 'residue_' if residue else ''

with open(f'../../golden_attention/amr/{file_prefix}amr_matrices.pkl', 'rb') as f:
    amr_connections = pickle.load(f)  # sentence, word, word

with pd.ExcelWriter(
        f'../../results/correlation/{model_name}/{model_size}/layer{model_layer}_{file_prefix}amr_{method}.xlsx') as writer_llama:
    llama_layer_attn = np.load(
        f'../../model_attention/amr/{model_name}/{model_size}/{file_prefix}amr_layer{model_layer}.npy')
    assert not np.isnan(np.sum(llama_layer_attn))

    for head in range(n_heads):
        df_llama = pd.DataFrame(columns=['r AMR', 'p AMR'])
        sheet_name = f'head {head}'
        llama_head_attn = llama_layer_attn[:, head, :, :]

        for senti in range(len(amr_connections)):
            if amr_connections[senti].shape[0] < 2:
                continue
            corrs_amr_llama = []
            r_amr = []

            mati = amr_connections[senti]
            llama_head_senti = llama_head_attn[senti]

            n_word = mati.shape[0]
            idx_tril = tril_idx(n_word)

            vec_llama = llama_head_senti[idx_tril]
            vec_amr = mati[idx_tril]

            corr_func = stats.pearsonr if method == 'pearson' else stats.spearmanr
            r_senti_amr, p_senti_amr = corr_func(vec_amr, vec_llama)
            if not np.isnan(r_senti_amr):
                r_amr.append(r_senti_amr)
                df_llama = df_llama.append(
                    {'r AMR': r_senti_amr, 'p AMR': p_senti_amr},
                    ignore_index=True)

        df_llama.to_excel(writer_llama, sheet_name=sheet_name, index=False)
