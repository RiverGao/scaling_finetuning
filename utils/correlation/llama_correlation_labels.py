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


model_name = 'gpt2'
model_size = 'large'
residue = True
head_pool = 'mean'
task = 'predicate'

# model_name = sys.argv[1]
# model_size = sys.argv[2]
# residue = eval(sys.argv[3])
# head_pool = sys.argv[4]
# task = sys.argv[5]

d_size_layer = {'7B': 32, '13B': 40, '30B': 60, 'large': 36}
n_layers = d_size_layer[model_size]
method = 'pearson'
np.random.seed(42)
file_prefix = 'residue_' if residue else ''

with open(f'../../golden_attention/reading_brain/{file_prefix}label_{task}.pkl', 'rb') as f:
    lbl_data = pickle.load(f)  # article, sentence, word, word

with pd.ExcelWriter(f'../../results/correlation/{model_name}/{model_size}/{head_pool}_{file_prefix}{task}_pearson.xlsx') as writer_model:
    with pd.ExcelWriter(f'../../results/correlation/{model_name}/{model_size}-random/{head_pool}_{file_prefix}{task}-random_pearson.xlsx') as writer_random:
        for layer in range(n_layers):
            df_model = pd.DataFrame(columns=[f'r {task}'])
            df_random = pd.DataFrame(columns=[f'r {task}'])
            sheet_name = f'layer {layer}'
            model_layer_attn = np.load(
                f'../../model_attention/reading_brain/{model_name}/{model_size}/p1/{file_prefix}rb_p1_layer{layer}.npy'
            )  # (n_arti, max_n_sents, n_head, max_sent_len, max_sent_len)
            assert not np.isnan(np.sum(model_layer_attn)), f'Layer {layer} has NaN values'

            for art_i in range(5):
                for sent_j in range(len(lbl_data[art_i])):
                    # print(f'layer {layer}, arti {art_i}, sent {sent_j}')
                    lbl_arti_sentj = np.array(lbl_data[art_i][sent_j])  # n_word * n_word

                    # average over attn heads
                    if head_pool == 'mean':
                        model_layer_arti_sentj = model_layer_attn[art_i][sent_j].mean(axis=0)
                    elif head_pool == 'max':
                        model_layer_arti_sentj = model_layer_attn[art_i][sent_j].max(axis=0)
                    else:
                        raise ValueError(f'Unknown head pooling: {head_pool}')

                    n_word = len(lbl_arti_sentj)
                    idx_tril = tril_idx(n_word)
                    # model_layer_arti_sentj = model_layer_arti_sentj[1:, 1:]

                    vec_model = model_layer_arti_sentj[idx_tril]
                    vec_random = vec_model.copy()
                    np.random.shuffle(vec_random)

                    vec_lbl = lbl_arti_sentj[idx_tril]

                    r_arti_sentj_lbl, _ = stats.pearsonr(vec_lbl, vec_model)
                    if r_arti_sentj_lbl == r_arti_sentj_lbl:
                        df_model = df_model.append(
                            {f'r {task}': r_arti_sentj_lbl},
                            ignore_index=True)

                    r_arti_sentj_lbl_random, _ = stats.pearsonr(vec_lbl, vec_random)
                    if r_arti_sentj_lbl_random == r_arti_sentj_lbl_random:
                        df_random = df_random.append(
                            {f'r {task}': r_arti_sentj_lbl_random},
                            ignore_index=True)

            df_model.to_excel(writer_model, sheet_name=sheet_name, index=False)
            df_random.to_excel(writer_random, sheet_name=sheet_name, index=False)
