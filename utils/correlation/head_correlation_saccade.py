import numpy as np
from scipy import stats
import pickle
import warnings
import pandas as pd
import sys
warnings.simplefilter(action='ignore')


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
model_layer = 45
p = 2
residue = True

# model_name = sys.argv[1]
# model_size = sys.argv[2]
# p = eval(sys.argv[3])
# model_residue = eval(sys.argv[4])
# data_residue = eval(sys.argv[5])
# head_pool = sys.argv[6]

d_size_layer = {'large': 36, '7B': 32, '13B': 40, '30B': 60}
d_size_heads = {'large': 20, '7B': 32, '13B': 40, '30B': 52}
n_layers = d_size_layer[model_size]
n_heads = d_size_heads[model_size]
method = 'pearson'

np.random.seed(42)
file_prefix = 'residue_' if residue else ''

with open(f'../../golden_attention/reading_brain/{file_prefix}saccade_num_p{p}.pkl', 'rb') as f:
    saccade_num = pickle.load(f)  # subj, article, sentence, (word, word)-array
with open(f'../../golden_attention/reading_brain/{file_prefix}saccade_dur_p{p}.pkl', 'rb') as f:
    saccade_dur = pickle.load(f)

with pd.ExcelWriter(
        f'../../results/correlation/{model_name}/{model_size}/layer{model_layer}_{file_prefix}sac_p{p}_pearson.xlsx'
) as writer_model:
    model_layer_attn = np.load(
        f'../../model_attention/reading_brain/{model_name}/{model_size}/p1/{file_prefix}rb_p1_layer{model_layer}.npy'
    )  # (n_arti, max_n_sents, n_head, max_sent_len, max_sent_len)
    assert not np.isnan(np.sum(model_layer_attn)), model_layer

    for head in range(n_heads):
        df_model = pd.DataFrame(columns=['subject', 'r number', 'r duration'])
        sheet_name = f'head {head}'
        model_head_attn = model_layer_attn[:, :, head, :, :]  # (n_arti, max_n_sents, max_sent_len, max_sent_len)

        for subj in range(len(saccade_num)):
            if p == 1 and subj == 0:
                continue  # L1_S01 is problematic
            s_num = saccade_num[subj]
            s_dur = saccade_dur[subj]
            if len(s_num) != 5:
                continue

            r_num = []
            r_dur = []
            for art_i in range(5):
                # if art_i >= len(act):
                #     continue
                for sent_j in range(len(s_num[art_i])):
                    # print(f'layer {layer}, subj {subj}, art {art_i}, sent {sent_j}')
                    s_num_arti_sentj = s_num[art_i][sent_j]
                    s_dur_arti_sentj = s_dur[art_i][sent_j]
                    model_head_arti_sentj = model_head_attn[art_i][sent_j]

                    n_word = s_num_arti_sentj.shape[0]
                    idx_tril = tril_idx(n_word)

                    vec_model = model_head_arti_sentj[idx_tril]

                    vec_num = s_num_arti_sentj[idx_tril]
                    vec_dur = s_dur_arti_sentj[idx_tril]

                    r_arti_sentj_num, _ = stats.pearsonr(vec_num, vec_model)
                    if not np.isnan(r_arti_sentj_num):
                        r_num.append(r_arti_sentj_num)

                    r_arti_sentj_dur, _ = stats.pearsonr(vec_dur, vec_model)
                    if not np.isnan(r_arti_sentj_dur):
                        r_dur.append(r_arti_sentj_dur)

            r_num_mean = np.mean(r_num)
            r_dur_mean = np.mean(r_dur)
            # print(f'subject {subj}:\n\tnumber r: {r_num_mean}\n\tduration r: {r_dur_mean}')
            df_model = df_model.append(
                {'subject': subj, 'r number': r_num_mean, 'r duration': r_dur_mean},
                ignore_index=True)

        df_model.to_excel(writer_model, sheet_name=sheet_name, index=False)
