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


model_name = 'llama'
model_size = '30B'
model_layer = 9
residue = True
head_pool = 'mean'
task = 'predicate'

# model_name = sys.argv[1]
# model_size = sys.argv[2]
# residue = eval(sys.argv[3])
# head_pool = sys.argv[4]
# task = sys.argv[5]

d_size_heads = {'7B': 32, '13B': 40, '30B': 52, 'large': 20}
n_heads = d_size_heads[model_size]
method = 'pearson'
np.random.seed(42)
file_prefix = 'residue_' if residue else ''

with open(f'../../golden_attention/reading_brain/{file_prefix}label_{task}.pkl', 'rb') as f:
    lbl_data = pickle.load(f)  # article, sentence, word, word

with pd.ExcelWriter(f'../../results/correlation/{model_name}/{model_size}/layer{model_layer}_{file_prefix}{task}_pearson.xlsx') as writer_model:
    model_layer_attn = np.load(
        f'../../model_attention/reading_brain/{model_name}/{model_size}/p1/{file_prefix}rb_p1_layer{model_layer}.npy'
    )  # (n_arti, max_n_sents, n_head, max_sent_len, max_sent_len)
    assert not np.isnan(np.sum(model_layer_attn))

    for head in range(n_heads):
        df_model = pd.DataFrame(columns=[f'r {task}'])
        sheet_name = f'head {head}'
        model_head_attn = model_layer_attn[:, :, head, :, :]

        for art_i in range(5):
            for sent_j in range(len(lbl_data[art_i])):
                # print(f'layer {layer}, arti {art_i}, sent {sent_j}')
                lbl_arti_sentj = np.array(lbl_data[art_i][sent_j])  # n_word * n_word

                # average over attn heads
                model_head_arti_sentj = model_head_attn[art_i][sent_j]

                n_word = len(lbl_arti_sentj)
                idx_tril = tril_idx(n_word)

                vec_model = model_head_arti_sentj[idx_tril]
                vec_lbl = lbl_arti_sentj[idx_tril]

                r_arti_sentj_lbl, _ = stats.pearsonr(vec_lbl, vec_model)
                if not np.isnan(r_arti_sentj_lbl):
                    df_model = df_model.append(
                        {f'r {task}': r_arti_sentj_lbl},
                        ignore_index=True)

        df_model.to_excel(writer_model, sheet_name=sheet_name, index=False)
