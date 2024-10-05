import numpy as np
import pickle

import scipy.stats
from sklearn.linear_model import LinearRegression
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


view = 'num'  # dur or num
p = 2  # 1 or 2
residue = True
region = 'angular'

file_prefix = 'residue_' if residue else ''
np.random.seed(42)

with open(f'../../golden_attention/reading_brain/roi_activations/L{p}/activations_{region}_p{p}.pkl', 'rb') as f:
    human_data = pickle.load(f)  # subj, article, sentence, (word, word)-array

with open(f'../../golden_attention/reading_brain/{file_prefix}mean_saccade_{view}_p{p}.pkl', 'rb') as f:
    mean_human_data = pickle.load(f)  # article, sentence, (word, word)-array

n_subjects = len(human_data)
sentence_lengths = [[] for i in range(5)]  # store the length of each sentence, list: article, sentence
r2_scores = []  # R^2 of LR for each subject

flatten_mean_human_data = []
for subject in range(n_subjects):
    flatten_subject_data = []  # length: 2-D, n_articles * n_sentences * n_words * (n_words + 1) / 2

    if p == 1 and subject == 0:
        continue  # L1_S01 is problematic
    print(f'subject {subject}')

    sub_data = human_data[subject]  # article, sentence, (word, word)-array
    if len(sub_data) != 5:
        print('\tskip')
        continue

    for arti in range(5):
        article_sub_data = sub_data[arti]
        n_sentences = len(article_sub_data)

        for sentj in range(n_sentences):
            sentence_sub_data = article_sub_data[sentj]  # (word, word)-array
            n_words = len(sentence_sub_data)

            # flattened indices
            tril_x, tril_y = tril_idx(n_words)

            if (p == 1 and subject == 1) or (p == 2 and subject == 0):
                # only store sentence lengths once, and reuse it afterwards
                sentence_lengths[arti].append(n_words)
                # flatten mean human data, only store them once
                sentence_mean_human_data = mean_human_data[arti][sentj]  # (word, word)-array
                flat_sentence_mean_human_data = sentence_mean_human_data[tril_x, tril_y].tolist()  # n_words * (n_words + 1) / 2
                flatten_mean_human_data.extend(flat_sentence_mean_human_data)

            # flatten human data
            flat_sentence_sub_data = sentence_sub_data[tril_x, tril_y].tolist()  # n_words * (n_words + 1) / 2
            flatten_subject_data.extend(flat_sentence_sub_data)

    # do regression for this subject
    X = np.array(flatten_mean_human_data).reshape(-1, 1)
    y = np.array(flatten_subject_data)
    reg_sub = LinearRegression().fit(X, y)
    score = reg_sub.score(X, y)
    r2_scores.append(score)

print(f'{view}, L{p}, noise ceiling: {np.mean(r2_scores)}, SD: {scipy.stats.sem(r2_scores)}')

# original
# num, L1, noise ceiling: 0.1559854152570265, SD: 0.007340818113794998
# num, L2, noise ceiling: 0.24632999658975888, SD: 0.008976470010783425
# dur, L1, noise ceiling: 0.13783828991106906, SD: 0.006340684317052503
# dur, L2, noise ceiling: 0.21455276206747248, SD: 0.008354765650394978

# residual
# num, L1, noise ceiling: 0.12004709688250854, SD: 0.005269349513039463
# num, L2, noise ceiling: 0.1802391293492779, SD: 0.006390994134367685
# dur, L1, noise ceiling: 0.11021495334933562, SD: 0.004989288819284424
# dur, L2, noise ceiling: 0.1655612299768772, SD: 0.00651107012216934





