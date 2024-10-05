import numpy as np
import pickle
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


def apply_values(values, target, size):
    # size is the width (also height) of the original value matrix, i.e. n_words
    it = iter(values)
    for i in range(size):
        for j in range(size):
            if j > i:
                break
            target[i, j] = next(it)


label_types = ['self', 'prev', 'start']
np.random.seed(42)

label_data = []  # type, article, sentence, word, word
for label_type in label_types:
    with open(f'../../golden_attention/amr/label_{label_type}.pkl', 'rb') as f:
        label_data.append(pickle.load(f))  # sentence, word, word

with open('../../golden_attention/amr/amr_matrices.pkl', 'rb') as f:
    amr_data = pickle.load(f)  # n_sentence of (word, word)-array

sentence_lengths = []  # store the length of each sentence, list: n_sentence
r2_score = 0  # R^2 of LR

flatten_label_data = []  # length: 2-D, n_sentences * n_words * (n_words + 1) / 2, n_labels
flatten_amr = []  # length: n_sentences * n_words * (n_words + 1) / 2

# flatten labels as well as amr matrices
# store lengths of all sentences for recovery
for sentj in range(len(amr_data)):
    print(f'sentence {sentj}')
    sentence_amr = amr_data[sentj]  # (n_words, n_words)-array
    n_words = len(sentence_amr)
    sentence_lengths.append(n_words)

    # flattened indices
    tril_x, tril_y = tril_idx(n_words)

    # flatten model attention
    flat_attn = sentence_amr[tril_x, tril_y].T.tolist()  # n_words * (n_words + 1) / 2
    flatten_amr.extend(flat_attn)  # as y in LR, n_sent * n_words * (n_words + 1) / 2

    # flatten labels
    flat_lbl_for_types = []  # n_labels, each element is array(n_words * (n_words + 1) / 2)
    for typek in range(len(label_types)):
        sentence_label = np.array(label_data[typek][sentj])  # array(n_words, n_words)
        flat_lbl_for_types.append(sentence_label[tril_x, tril_y])  # append an array(n_words * (n_words + 1) / 2)
    flat_lbl = np.stack(flat_lbl_for_types).T.tolist()  # n_words * (n_words + 1) / 2, n_labels
    flatten_label_data.extend(flat_lbl)


X = np.array(flatten_label_data)
y = np.array(flatten_amr)
reg = LinearRegression().fit(X, y)
score = reg.score(X, y)

# get residue
flatten_prediction = reg.predict(flatten_label_data)  # array(n_arti * n_sent * n_words * (n_words + 1) / 2)
flatten_residue = y - flatten_prediction  # array(n_arti * n_sent * n_words * (n_words + 1) / 2)
start_pos = 0  # pointer for copying residue values

# recover residue to the original format: # n_sents, n_words, n_words
for sentj in range(len(sentence_lengths)):
    n_words = sentence_lengths[sentj]  # number of words in this sentence
    n_values = n_words * (n_words + 1) // 2  # number of values in the lower triangle matrix
    values_to_apply = flatten_residue[start_pos: start_pos + n_values]
    target = amr_data[sentj]
    target.dtype = np.float64
    apply_values(values_to_apply, amr_data[sentj], n_words)
    start_pos += n_values


r2_score = score


print(r2_score)
with open('../../golden_attention/amr/residue_amr_matrices.pkl', 'wb') as f:
    pickle.dump(amr_data, f)
