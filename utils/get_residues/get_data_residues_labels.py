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


# view = sys.argv[1]  # duration or number
# p = eval(sys.argv[2])  # 1 or 2

task = 'attribute'  # predicate or attribute

label_types = ['self', 'prev', 'start']
d_size_layers = {'7B': 32, '13B': 40, '7Bhf': 32, '13Bhf': 40}
d_size_heads = {'7B': 32, '13B': 40, '7Bhf': 32, '13Bhf': 40}
np.random.seed(42)

label_data = []  # trivial labels, type, article, sentence, word, word
for label_type in label_types:
    with open(f'../../golden_attention/reading_brain/label_{label_type}.pkl', 'rb') as f:
        label_data.append(pickle.load(f))  # article, sentence, word, word

with open(f'../../golden_attention/reading_brain/label_{task}.pkl', 'rb') as f:
    neural_data = pickle.load(f)  # article, sentence, word, word

sentence_lengths = [[] for i in range(5)]  # store the length of each sentence, list: article, sentence
r2_scores = 0  # R^2 of LR


flatten_label_data = []  # length: 2-D, n_articles * n_sentences * n_words * (n_words + 1) / 2, n_labels
flatten_neural = []  # length: 2-D, n_articles * n_sentences * n_words * (n_words + 1) / 2


# flatten labels as well as model attentions
# store lengths of all sentences for recovery
for arti in range(5):
    # print(f'article {arti}')
    sentence_data0_in_article = label_data[0][arti]  # n_sent, n_words, n_words
    for sentj in range(len(sentence_data0_in_article)):
        # print(f'sentence {sentj}')
        sentence_data0 = np.array(sentence_data0_in_article[sentj])  # n_words, n_words
        n_words = len(sentence_data0)
        sentence_lengths[arti].append(n_words)

        # flattened indices
        tril_x, tril_y = tril_idx(n_words)

        # flatten model attention
        model_layer_arti_sentj = np.array(neural_data[arti][sentj])  # (n_words, n_words)-array
        flat_attn = model_layer_arti_sentj[tril_x, tril_y].T.tolist()  # n_words * (n_words + 1) / 2
        flatten_neural.extend(flat_attn)  # as y in LR, n_arti * n_sent * n_words * (n_words + 1) / 2

        # flatten labels
        flat_lbl_for_types = []  # n_labels, each element is array(n_words * (n_words + 1) / 2)
        for typek in range(len(label_types)):
            sentence_data = np.array(label_data[typek][arti][sentj])  # array(n_words, n_words)
            flat_lbl_for_types.append(sentence_data[tril_x, tril_y])  # append an array(n_words * (n_words + 1) / 2)
        flat_lbl = np.stack(flat_lbl_for_types).T.tolist()  # n_words * (n_words + 1) / 2, n_labels
        flatten_label_data.extend(flat_lbl)


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
        apply_values(values_to_apply, neural_data[arti][sentj], n_words)
        start_pos += n_values

r2_scores = score


print(np.mean(r2_scores))
with open(f'../../golden_attention/reading_brain/residue_label_{task}.pkl', 'wb') as f:
    pickle.dump(neural_data, f)
np.save(f'../../results/lr_scores/label/scores_{task}.npy', np.array(r2_scores))
