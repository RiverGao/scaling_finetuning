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


# model_name = 'gpt2'
# model_size = 'large'
# p = 2
# model_residue = True
# data_residue = True
# head_pool = 'mean'

model_name = sys.argv[1]
model_size = sys.argv[2]
p = eval(sys.argv[3])
model_residue = eval(sys.argv[4])
data_residue = eval(sys.argv[5])
head_pool = sys.argv[6]

d_size_layer = {'large': 36, '7B': 32, '13B': 40, '30B': 60}
n_layers = d_size_layer[model_size]
method = 'pearson'

np.random.seed(42)
model_prefix = 'residue_' if model_residue else ''
data_prefix = 'residue_' if data_residue else ''

with open(f'../../golden_attention/reading_brain/{data_prefix}saccade_num_p{p}.pkl', 'rb') as f:
    saccade_num = pickle.load(f)  # subj, article, sentence, (word, word)-array
with open(f'../../golden_attention/reading_brain/{data_prefix}saccade_dur_p{p}.pkl', 'rb') as f:
    saccade_dur = pickle.load(f)

with pd.ExcelWriter(
        f'../../results/correlation/{model_name}/{model_size}/{head_pool}_{model_prefix}m_{data_prefix}s_sac_p{p}_pearson.xlsx'
) as writer_model:
    with pd.ExcelWriter(
            f'../../results/correlation/{model_name}/{model_size}-random/{head_pool}_{model_prefix}m_{data_prefix}s_sac_p{p}-random_pearson.xlsx'
    ) as writer_random:
        for layer in range(n_layers):
            df_model = pd.DataFrame(columns=['subject', 'r number', 'r duration'])
            df_random = pd.DataFrame(columns=['subject', 'r number', 'r duration'])
            sheet_name = f'layer {layer}'
            model_layer_attn = np.load(
                f'../../model_attention/reading_brain/{model_name}/{model_size}/p1/{model_prefix}rb_p1_layer{layer}.npy'
            )  # (n_arti, max_n_sents, n_head, max_sent_len, max_sent_len)
            assert not np.isnan(np.sum(model_layer_attn)), layer
            for subj in range(len(saccade_num)):
                if p == 1 and subj == 0:
                    continue  # L1_S01 is problematic
                s_num = saccade_num[subj]
                s_dur = saccade_dur[subj]
                if len(s_num) != 5:
                    continue

                r_num = []
                r_dur = []
                r_num_random = []
                r_dur_random = []
                for art_i in range(5):
                    # if art_i >= len(act):
                    #     continue
                    for sent_j in range(len(s_num[art_i])):
                        # print(f'layer {layer}, subj {subj}, art {art_i}, sent {sent_j}')
                        s_num_arti_sentj = s_num[art_i][sent_j]
                        s_dur_arti_sentj = s_dur[art_i][sent_j]
                        # average over attn heads
                        if head_pool == 'mean':
                            model_layer_arti_sentj = model_layer_attn[art_i][sent_j].mean(axis=0)
                        elif head_pool == 'max':
                            model_layer_arti_sentj = model_layer_attn[art_i][sent_j].max(axis=0)
                        else:
                            raise ValueError(f'Unknown head pooling: {head_pool}')
                        # model_layer_arti_sentj = model_layer_attn[art_i][sent_j]

                        n_word = s_num_arti_sentj.shape[0]
                        idx_tril = tril_idx(n_word)
                        # model_layer_arti_sentj = model_layer_arti_sentj[1:, 1:]
                        model_layer_arti_sentj = model_layer_arti_sentj

                        vec_model = model_layer_arti_sentj[idx_tril]
                        vec_random = vec_model.copy()
                        np.random.shuffle(vec_random)

                        vec_num = s_num_arti_sentj[idx_tril]
                        vec_dur = s_dur_arti_sentj[idx_tril]

                        r_arti_sentj_num, _ = stats.pearsonr(vec_num, vec_model)
                        if r_arti_sentj_num == r_arti_sentj_num:
                            r_num.append(r_arti_sentj_num)

                        r_arti_sentj_dur, _ = stats.pearsonr(vec_dur, vec_model)
                        if r_arti_sentj_dur == r_arti_sentj_dur:
                            r_dur.append(r_arti_sentj_dur)

                        r_arti_sentj_num_random, _ = stats.pearsonr(vec_num, vec_random)
                        if r_arti_sentj_num_random == r_arti_sentj_num_random:
                            r_num_random.append(r_arti_sentj_num_random)

                        r_arti_sentj_dur_random, _ = stats.pearsonr(vec_dur, vec_random)
                        if r_arti_sentj_dur_random == r_arti_sentj_dur_random:
                            r_dur_random.append(r_arti_sentj_dur_random)

                r_num_mean = np.mean(r_num)
                r_dur_mean = np.mean(r_dur)
                r_num_random_mean = np.mean(r_num_random)
                r_dur_random_mean = np.mean(r_dur_random)
                # print(f'subject {subj}:\n\tnumber r: {r_num_mean}\n\tduration r: {r_dur_mean}')
                df_model = df_model.append(
                    {'subject': subj, 'r number': r_num_mean, 'r duration': r_dur_mean},
                    ignore_index=True)
                df_random = df_random.append(
                    {'subject': subj, 'r number': r_num_random_mean, 'r duration': r_dur_random_mean},
                    ignore_index=True)

            df_model.to_excel(writer_model, sheet_name=sheet_name, index=False)
            df_random.to_excel(writer_random, sheet_name=sheet_name, index=False)
