import numpy as np
from scipy import stats
import pickle
import warnings
import sys
warnings.simplefilter(action='ignore')


p = 2
region = 'anterior'
residue = False

# p = eval(sys.argv[1])
# region = sys.argv[2]
# residue = eval(sys.argv[3])

problems = {1: [0, 6, 20, 31, 47, 49, 50, 51], 2: [27, 41, 42, 44, 47, 54]}  # problematic subject indices
file_prefix = 'residue_' if residue else ''


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


# with open(f'/home/river/Workbench/datasets/reading_brain_fmri/L{p}/{file_prefix}activations_{region}_p{p}.pkl', 'rb') as f:
with open(f'../../golden_attention/reading_brain/roi_activations/L{p}/{file_prefix}activations_{region}_p{p}.pkl', 'rb') as f:
    activations = pickle.load(f)  # subj, article, sentence, (word, word)-array


# first obtain the mean saccade matrices
mean_activations = []  # size expected to be: article, sentence, (word, word)-array
subj_counted = 0  # used to calculate the mean
for subj in range(len(activations)):
    if subj in problems[p]:
        continue  # skip problematic subjects

    act = activations[subj]  # article, sentence, (word, word)-array
    if len(act) != 5:
        continue

    # valid saccade data, can be used to initialize mean_s_num/dur
    if not mean_activations:
        mean_activations = act
        subj_counted += 1
        continue

    # if it is after initialization, add this subject's each (word, word)-array to the mean
    # needed to iterate over articles and sentences
    for art_i in range(5):
        for sent_j in range(len(act[art_i])):
            act_arti_sentj = np.array(act[art_i][sent_j])  # (word, word)-array
            assert not np.isnan(act_arti_sentj).any(), f'subj {subj}, {art_i}-{sent_j}'

            # validate format
            for aas in [act_arti_sentj]:
                assert type(aas) == np.ndarray
                assert aas.shape[0] == aas.shape[1]

            # add the 2 saccade matrices to the means
            # use a mean-while-adding method
            # if there are already n samples and a mean, and a new sample x comes
            # then the new mean should be mean * n / (n+1) + x / (n+1)
            mean_activations[art_i][sent_j] = np.array(mean_activations[art_i][sent_j])
            mean_activations[art_i][sent_j] = mean_activations[art_i][sent_j] * subj_counted / (subj_counted + 1) + act_arti_sentj / (subj_counted + 1)
    # after iteration, add 1 to the number of counted subjects
    subj_counted += 1

# after iterating over all subjects
# now mean_s_num, mean_s_dur are obtained, and the next step is to calculate correlation
# do the iteration again
r_act = []  # size expected to be: subject
for subj in range(len(activations)):
    if subj in problems[p]:
        continue  # skip problematic subjects

    act = activations[subj]  # article, sentence, (word, word)-array
    if len(act) != 5:
        continue

    r_act_subj = []  # size expected to be: article * sentences
    sent_counted = 0  # used to validate r_num/dur_subj format
    for art_i in range(5):
        for sent_j in range(len(act[art_i])):
            # 4 (word, word)-arrays
            act_arti_sentj = np.array(act[art_i][sent_j])
            mean_act_arti_sentj = mean_activations[art_i][sent_j]

            # validate format
            for aas in [act_arti_sentj, mean_act_arti_sentj]:
                assert type(aas) == np.ndarray
                assert aas.shape[0] == aas.shape[1]

            # lower triangle indices
            n_word = act_arti_sentj.shape[0]
            idx_tril = tril_idx(n_word)

            # vectorize the (word, word)-arrays
            vec_act = act_arti_sentj[idx_tril]
            vec_mean_act = mean_act_arti_sentj[idx_tril]

            # calculate correlations
            r_arti_sentj_num, _ = stats.pearsonr(vec_act, vec_mean_act)
            if not np.isnan(r_arti_sentj_num):
                r_act_subj.append(r_arti_sentj_num)
                sent_counted += 1

    # after iterating over all sentences in all articles for this subject
    # append the mean correlation to r_num/dur (size = n_subj)
    # assert len(r_num_subj) == len(r_dur_subj) == sent_counted
    r_act.append(np.mean(r_act_subj))

# after iterating over all subjects
# assert (p == 1 and len(r_num) == 51) or (p == 2 and len(r_num) == 54)
r_act_mean = np.mean(r_act)  # scalar, noise ceiling

print(f'L{p}, residue {residue}, region {region}: {r_act_mean}')

# original
# L1:
#    middle: 0.1422154901638094
#    inferior: 0.14655120369083222
#    superior: 0.13922533179913824
#    angular: 0.15382562383967927
# L2:
#    middle: 0.13910987315438422
#    inferior: 0.1404541365418864
#    superior: 0.12844558073711515
#    angular: 0.14174022251211715

# residual
# L1:
#     angular:
#     middle:
#     superior:
#     inferior:
#     anterior:
# L2:
#     angular:
#     middle:
#     superior:
#     inferior:
#     anterior:
