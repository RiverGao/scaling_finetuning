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


model_name = sys.argv[1]
model_size = sys.argv[2]
p = eval(sys.argv[3])
model_residue = eval(sys.argv[4])
data_residue = eval(sys.argv[5])
head_pool = sys.argv[6]
region = sys.argv[7]  # brain region

# model_name = 'llama'
# model_size = '7B'
# p = 2
# model_residue = False
# data_residue = True
# head_pool = 'mean'
# region = 'angular'  # brain region

d_size_layer = {'7B': 32, '13B': 40, '33B': 56}
n_layers = d_size_layer[model_size]
method = 'pearson'
problems = {1: [0, 6, 20, 31, 47, 49, 50, 51], 2: [27, 41, 42, 44, 47, 54]}  # problematic subject indices

np.random.seed(42)
model_prefix = 'residue_' if model_residue else ''
data_prefix = 'residue_' if data_residue else ''


with open(f'/home/river/Workbench/datasets/reading_brain_fmri/L{p}/{data_prefix}activations_{region}_p{p}.pkl', 'rb') as f:
    activations = pickle.load(f)  # subj, article, sentence, word, word

with pd.ExcelWriter(
        f'../../results/correlation/{model_name}/{model_size}/{head_pool}_{model_prefix}m_{data_prefix}f_fmri_{region}_p{p}_pearson.xlsx'
) as writer_model:
    with pd.ExcelWriter(
            f'../../results/correlation/{model_name}/{model_size}-random/{head_pool}_{model_prefix}m_{data_prefix}f_fmri_{region}_p{p}-random_pearson.xlsx'
    ) as writer_random:
        for layer in range(n_layers):
            # df_model = pd.DataFrame(columns=['subject', 'r activation'])
            # df_random = pd.DataFrame(columns=['subject', 'r activation'])
            l_model = []
            l_random = []
            sheet_name = f'layer {layer}'
            model_layer_attn = np.load(
                f'../../model_attention/reading_brain/{model_name}/{model_size}/p1/{model_prefix}rb_p1_layer{layer}.npy'
            )  # (n_arti, max_n_sents, n_head, max_sent_len, max_sent_len)
            assert not np.isnan(np.sum(model_layer_attn)), layer
            for subj in range(len(activations)):
                if subj in problems[p]:
                    continue  # skip problematic subjects
                act = activations[subj]
                if len(act) != 5:
                    continue

                r_act = []
                r_act_random = []
                for art_i in range(5):
                    # if art_i >= len(act):
                    #     continue
                    for sent_j in range(len(act[art_i])):
                        # print(f'layer {layer}, subj {subj}, art {art_i}, sent {sent_j}')
                        act_arti_sentj = np.array(act[art_i][sent_j])
                        # average over attn heads
                        if head_pool == 'mean':
                            model_layer_arti_sentj = model_layer_attn[art_i][sent_j].mean(axis=0)
                        elif head_pool == 'max':
                            model_layer_arti_sentj = model_layer_attn[art_i][sent_j].max(axis=0)
                        else:
                            raise ValueError(f'Unknown head pooling: {head_pool}')
                        # model_layer_arti_sentj = model_layer_attn[art_i][sent_j]

                        n_word = act_arti_sentj.shape[0]
                        idx_tril = tril_idx(n_word)
                        # model_layer_arti_sentj = model_layer_arti_sentj[1:, 1:]
                        model_layer_arti_sentj = model_layer_arti_sentj

                        vec_model = model_layer_arti_sentj[idx_tril]
                        vec_random = vec_model.copy()
                        np.random.shuffle(vec_random)

                        vec_act = act_arti_sentj[idx_tril]

                        r_act_arti_sentj, _ = stats.pearsonr(vec_act, vec_model)
                        if r_act_arti_sentj == r_act_arti_sentj:
                            r_act.append(r_act_arti_sentj)

                        r_act_arti_sentj_random, _ = stats.pearsonr(vec_act, vec_random)
                        if r_act_arti_sentj_random == r_act_arti_sentj_random:
                            r_act_random.append(r_act_arti_sentj_random)

                r_act_mean = np.mean(r_act)
                r_act_random_mean = np.mean(r_act_random)
                # print(f'subject {subj}:\n\tnumber r: {r_num_mean}\n\tduration r: {r_dur_mean}')
                l_model.append({'subject': subj, 'r activation': r_act_mean})
                l_random.append({'subject': subj, 'r activation': r_act_random_mean})

            df_model = pd.DataFrame(l_model)
            df_random = pd.DataFrame(l_random)
            df_model.to_excel(writer_model, sheet_name=sheet_name, index=False)
            df_random.to_excel(writer_random, sheet_name=sheet_name, index=False)
