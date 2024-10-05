import numpy as np
from scipy import stats
import pickle
import warnings


p = 2
data_residue = True

data_prefix = 'residue_' if data_residue else ''

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


with open(f'../../golden_attention/reading_brain/{data_prefix}saccade_num_p{p}.pkl', 'rb') as f:
    saccade_num = pickle.load(f)  # subj, article, sentence, (word, word)-array
with open(f'../../golden_attention/reading_brain/{data_prefix}saccade_dur_p{p}.pkl', 'rb') as f:
    saccade_dur = pickle.load(f)  # subj, article, sentence, (word, word)-array


# first obtain the mean saccade matrices
mean_s_num = []  # size expected to be: article, sentence, (word, word)-array
mean_s_dur = []
subj_counted = 0  # used to calculate the mean
for subj in range(len(saccade_num)):
    if p == 1 and subj == 0:
        continue  # L1_S01 is problematic

    s_num = saccade_num[subj]  # article, sentence, (word, word)-array
    s_dur = saccade_dur[subj]
    if len(s_num) != 5:
        continue

    # valid saccade data, can be used to initialize mean_s_num/dur
    if not mean_s_num:
        assert not mean_s_dur
        mean_s_num = s_num
        mean_s_dur = s_dur
        subj_counted += 1
        continue

    # if it is after initialization, add this subject's each (word, word)-array to the mean
    # needed to iterate over articles and sentences
    for art_i in range(5):
        for sent_j in range(len(s_num[art_i])):
            s_num_arti_sentj = s_num[art_i][sent_j]  # (word, word)-array
            s_dur_arti_sentj = s_dur[art_i][sent_j]

            # validate format
            for sas in [s_num_arti_sentj, s_dur_arti_sentj]:
                assert type(sas) == np.ndarray
                assert sas.shape[0] == sas.shape[1]

            # add the 2 saccade matrices to the means
            # use a mean-while-adding method
            # if there are already n samples and a mean, and a new sample x comes
            # then the new mean should be mean * n / (n+1) + x / (n+1)
            mean_s_num[art_i][sent_j] = mean_s_num[art_i][sent_j] * subj_counted / (subj_counted + 1) + s_num_arti_sentj / (subj_counted + 1)
            mean_s_dur[art_i][sent_j] = mean_s_dur[art_i][sent_j] * subj_counted / (subj_counted + 1) + s_dur_arti_sentj / (subj_counted + 1)
    # after iteration, add 1 to the number of counted subjects
    subj_counted += 1

# with open(f'{data_prefix}mean_saccade_num_p{p}.pkl', 'wb') as f:
#     pickle.dump(mean_s_num, f)
# with open(f'{data_prefix}mean_saccade_dur_p{p}.pkl', 'wb') as f:
#     pickle.dump(mean_s_dur, f)
# assert 0

# after iterating over all subjects
# now mean_s_num, mean_s_dur are obtained, and the next step is to calculate correlation
# do the iteration again
r_num = []  # size expected to be: subject
r_dur = []
for subj in range(len(saccade_num)):
    if p == 1 and subj == 0:
        continue  # L1_S01 is problematic

    s_num = saccade_num[subj]  # article, sentence, (word, word)-array
    s_dur = saccade_dur[subj]
    if len(s_num) != 5:
        continue

    r_num_subj = []  # size expected to be: article * sentences
    r_dur_subj = []
    sent_counted = 0  # used to validate r_num/dur_subj format
    for art_i in range(5):
        for sent_j in range(len(s_num[art_i])):
            sent_counted += 1
            # 4 (word, word)-arrays
            s_num_arti_sentj = s_num[art_i][sent_j]
            s_dur_arti_sentj = s_dur[art_i][sent_j]
            mean_s_num_arti_sentj = mean_s_num[art_i][sent_j]
            mean_s_dur_arti_sentj = mean_s_dur[art_i][sent_j]

            # validate format
            for sas in [s_num_arti_sentj, s_dur_arti_sentj, mean_s_num_arti_sentj, mean_s_dur_arti_sentj]:
                assert type(sas) == np.ndarray
                assert sas.shape[0] == sas.shape[1]

            # lower triangle indices
            n_word = s_num_arti_sentj.shape[0]
            idx_tril = tril_idx(n_word)

            # vectorize the (word, word)-arrays
            vec_num = s_num_arti_sentj[idx_tril]
            vec_dur = s_dur_arti_sentj[idx_tril]
            vec_mean_num = mean_s_num_arti_sentj[idx_tril]
            vec_mean_dur = mean_s_dur_arti_sentj[idx_tril]

            # calculate correlations
            r_arti_sentj_num, _ = stats.pearsonr(vec_num, vec_mean_num)
            if r_arti_sentj_num == r_arti_sentj_num:
                r_num_subj.append(r_arti_sentj_num)

            r_arti_sentj_dur, _ = stats.pearsonr(vec_dur, vec_mean_dur)
            if r_arti_sentj_dur == r_arti_sentj_dur:
                r_dur_subj.append(r_arti_sentj_dur)

    # after iterating over all sentences in all articles for this subject
    # append the mean correlation to r_num/dur (size = n_subj)
    # assert len(r_num_subj) == len(r_dur_subj) == sent_counted
    r_num.append(np.mean(r_num_subj))
    r_dur.append(np.mean(r_dur_subj))

# after iterating over all subjects
assert (p == 1 and len(r_num) == len(r_dur) == 51) or (p == 2 and len(r_num) == len(r_dur) == 54)
r_num_mean = np.mean(r_num)  # scalar, noise ceiling
r_dur_mean = np.mean(r_dur)

print(f'L{p}, residue {data_residue}, noise ceiling:\n{r_num_mean} for number\n{r_dur_mean} for duration')


# L1, residue True, noise ceiling:
# 0.3364847242575918 for number
# 0.3245498953577841 for duration
# L2, residue True, noise ceiling:
# 0.41761399245058306 for number
# 0.40640372302894495 for duration


