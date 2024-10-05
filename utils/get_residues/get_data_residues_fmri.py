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
            target[i][j] = next(it)


# p = 1  # 1 or 2
# region = 'angular'
p = eval(sys.argv[1])  # 1 or 2
region = sys.argv[2]
assert region in ['angular', 'middle', 'superior', 'inferior', 'anterior']

label_types = ['self', 'prev', 'start']
d_size_layers = {'7B': 32, '13B': 40, '7Bhf': 32, '13Bhf': 40}
d_size_heads = {'7B': 32, '13B': 40, '7Bhf': 32, '13Bhf': 40}
np.random.seed(42)

label_data = []  # type, article, sentence, word, word
for label_type in label_types:
    with open(f'../../golden_attention/reading_brain/label_{label_type}.pkl', 'rb') as f:
        label_data.append(pickle.load(f))  # article, sentence, word, word

with open(f'/home/river/Workbench/datasets/reading_brain_fmri/L{p}/activations_{region}_p{p}.pkl', 'rb') as f:
    neural_data = pickle.load(f)  # subj, article, sentence, word, word

sentence_lengths = [[] for i in range(5)]  # store the length of each sentence, list: article, sentence
r2_scores = []  # R^2 of LR for each subject
n_subjects = len(neural_data)
problems = {1: [0, 6, 20, 31, 47, 49, 50, 51], 2: [27, 41, 42, 44, 47, 54]}  # problematic subject indices

for isub in range(n_subjects):
    # do linear regression for each subject independently
    # output residues of each layer
    # print(f'subject {isub}')
    if isub in problems[p]:
        continue  # skip problematic subjects

    flatten_label_data = []  # length: 2-D, n_articles * n_sentences * n_words * (n_words + 1) / 2, n_labels
    flatten_neural = []  # length: 2-D, n_articles * n_sentences * n_words * (n_words + 1) / 2

    # read model attentions in this layer
    sub_neural = neural_data[isub]  # article, sentence, word, word
    # assert not np.isnan(np.sum(sub_neural)), f'Subject {isub} has NaN values'
    if len(sub_neural) != 5:
        # print('skip')
        continue

    # flatten labels as well as model attentions
    # store lengths of all sentences for recovery
    for arti in range(5):
        # print(f'article {arti}')
        sentence_data0_in_article = label_data[0][arti]  # n_sent, n_words, n_words
        for sentj in range(len(sentence_data0_in_article)):
            # print(f'sentence {sentj}')
            sentence_data0 = np.array(sentence_data0_in_article[sentj])  # n_words, n_words
            n_words = len(sentence_data0)
            if (p == 1 and isub == 1) or (p == 2 and isub == 0):
                # only store sentence lengths once, and reuse it afterwards
                sentence_lengths[arti].append(n_words)

            # flattened indices
            tril_x, tril_y = tril_idx(n_words)

            # flatten model attention
            model_layer_arti_sentj = np.array(sub_neural[arti][sentj])  # (n_words, n_words)-array
            flat_attn = model_layer_arti_sentj[tril_x, tril_y].T.tolist()  # n_words * (n_words + 1) / 2
            flatten_neural.extend(flat_attn)  # as y in LR, n_arti * n_sent * n_words * (n_words + 1) / 2

            # flatten labels
            flat_lbl_for_types = []  # n_labels, each element is array(n_words * (n_words + 1) / 2)
            for typek in range(len(label_types)):
                sentence_data = np.array(label_data[typek][arti][sentj])  # array(n_words, n_words)
                flat_lbl_for_types.append(sentence_data[tril_x, tril_y])  # append an array(n_words * (n_words + 1) / 2)
            flat_lbl = np.stack(flat_lbl_for_types).T.tolist()  # n_words * (n_words + 1) / 2, n_labels
            flatten_label_data.extend(flat_lbl)

    # recover attention structure by adding sentence attentions on the zeros_like matrix
    # residue_neural = sub_neural  # article, sentence, (word, word)-array

    X = flatten_label_data
    y = np.array(flatten_neural)
    reg_sub = LinearRegression().fit(X, y)
    score = reg_sub.score(X, y)

    # get residue
    flatten_sub_prediction = reg_sub.predict(flatten_label_data)  # array(n_arti * n_sent * n_words * (n_words + 1) / 2)
    flatten_sub_residue = y - flatten_sub_prediction  # array(n_arti * n_sent * n_words * (n_words + 1) / 2)
    start_pos = 0  # pointer for copying residue values

    # recover residue to the original format: # (n_arti, max_n_sents, n_head, max_sent_len, max_sent_len)
    for arti in range(5):
        for sentj in range(len(sentence_lengths[arti])):
            n_words = sentence_lengths[arti][sentj]  # number of words in this sentence
            n_values = n_words * (n_words + 1) // 2  # number of values in the lower triangle matrix
            values_to_apply = flatten_sub_residue[start_pos: start_pos + n_values]
            apply_values(values_to_apply, sub_neural[arti][sentj], n_words)
            start_pos += n_values

    r2_scores.append(score)


print(np.mean(r2_scores))
with open(f'/home/river/Workbench/datasets/reading_brain_fmri/L{p}/residue_activations_{region}_p{p}.pkl', 'wb') as f:
    pickle.dump(neural_data, f)
np.save(f'../../results/lr_scores/fmri/scores_{region}_p{p}.npy', np.array(r2_scores))
