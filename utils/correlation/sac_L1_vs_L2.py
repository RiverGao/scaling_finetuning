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


data_residue = False
method = 'pearson'

np.random.seed(42)
data_prefix = 'residue_' if data_residue else ''

with open(f'../../golden_attention/reading_brain/{data_prefix}mean_saccade_num_p1.pkl', 'rb') as f:
    saccade_num_1 = pickle.load(f)  # article, sentence, (word, word)-array
with open(f'../../golden_attention/reading_brain/{data_prefix}mean_saccade_num_p2.pkl', 'rb') as f:
    saccade_num_2 = pickle.load(f)  # article, sentence, (word, word)-array

# r_num = []
# r_dur = []
# for art_i in range(5):
#     assert len(saccade_num_1) == len(saccade_num_2)
#     for sent_j in range(len(saccade_num_1[art_i])):
#         # print(f'layer {layer}, subj {subj}, art {art_i}, sent {sent_j}')
#         s1_arti_sentj = saccade_num_1[art_i][sent_j]  # (word, word)
#         s2_arti_sentj = saccade_num_2[art_i][sent_j]
#
#         n_word = s1_arti_sentj.shape[0]
#         idx_tril = tril_idx(n_word)
#
#         vec_s1 = s1_arti_sentj[idx_tril]
#         vec_s2 = s2_arti_sentj[idx_tril]
#
#         rs12, _ = stats.pearsonr(vec_s1, vec_s2)
#         if not np.isnan(rs12).any():
#             r_num.append(rs12)
# r_num_mean = np.mean(r_num)

vec_s1 = []
vec_s2 = []
for art_i in range(5):
    assert len(saccade_num_1) == len(saccade_num_2)
    for sent_j in range(len(saccade_num_1[art_i])):
        # print(f'layer {layer}, subj {subj}, art {art_i}, sent {sent_j}')
        s1_arti_sentj = saccade_num_1[art_i][sent_j]  # (word, word)
        s2_arti_sentj = saccade_num_2[art_i][sent_j]

        n_word = s1_arti_sentj.shape[0]
        idx_tril = tril_idx(n_word)

        vec_s1.extend(s1_arti_sentj[idx_tril])
        vec_s2.extend(s2_arti_sentj[idx_tril])

r_num_mean, _ = stats.pearsonr(vec_s1, vec_s2)

print(r_num_mean)
