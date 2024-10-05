import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
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


dataset = 'GeCo'
view = 'num'  # dur or num
p = 2  # 1 or 2
model_name = 'alpaca'
model_size = '13B'
residue = False
instruct = False
attn_method = ''

d_size_layers = {'large': 36, '7B': 32, '13B': 40, '30B': 60, '65B': 80}
d_size_heads = {'large': 20, '7B': 32, '13B': 40, '30B': 52, '65B': 64}
np.random.seed(42)
file_prefix = 'residue_' if residue else ''
instr_prefix = 'instr_' if instruct else ''
n_layers = d_size_layers[model_size]
n_heads = 1 if attn_method else d_size_heads[model_size]

with open(f'../../golden_attention/{dataset}/{file_prefix}saccade_{view}_p{p}.pkl', 'rb') as f:
    human_data = pickle.load(f)  # subj, article, sentence, (word, word)-array

n_subjects = len(human_data)
n_articles = 5 if dataset == 'reading_brain' else 2
sentence_lengths = [[] for i in range(n_articles)]  # store the length of each sentence, list: article, sentence
all_r2_scores = []  # R^2 of LR for each layer, each subject

for layer in range(n_layers):
    layer_r2_scores = []  # R^2 of LR for each subject in this layer

    dataset_code = 'rb' if dataset == 'reading_brain' else 'gc'
    model_layer_attn = np.load(
        f'../../model_attention/{dataset}/{model_name}/{model_size}/{attn_method if attn_method else "p1"}/{file_prefix}{instr_prefix}{dataset_code}_p1_layer{layer}.npy'
    )  # (n_arti, max_n_sents, n_head, max_sent_len, max_sent_len)
    assert not np.isnan(np.sum(model_layer_attn)), f'Layer {layer} has NaN values'
    flatten_model_attn = [[] for i in range(n_heads)]  # length: 2-D, n_articles * n_sentences * n_words * (n_words + 1) / 2, n_heads

    for subject in range(n_subjects):
        flatten_subject_data = []  # length: 2-D, n_articles * n_sentences * n_words * (n_words + 1) / 2

        if dataset == "reading_brain" and p == 1 and subject == 0:
            continue  # L1_S01 is problematic
        print(f'layer {layer}, subject {subject}')

        sub_data = human_data[subject]  # article, sentence, (word, word)-array
        if len(sub_data) != n_articles:
            print('\tskip')
            continue

        for arti in range(n_articles):
            article_sub_data = sub_data[arti]
            n_sentences = len(article_sub_data)

            for sentj in range(n_sentences):
                sentence_sub_data = article_sub_data[sentj]  # (word, word)-array
                n_words = len(sentence_sub_data)

                # flattened indices
                tril_x, tril_y = tril_idx(n_words)

                if (dataset == 'reading_brain' and p == 1 and subject == 1) or (dataset == 'reading_brain' and p == 2 and subject == 0) or (dataset == 'GeCo' and subject == 0):
                    # only store sentence lengths once, and reuse it afterwards
                    sentence_lengths[arti].append(n_words)
                    # flatten attentions in each head, only store them once
                    for head in range(n_heads):
                        sentence_head_attn = model_layer_attn[arti, sentj, head, :, :]  # (max_sent_len, max_sent_len)
                        # sentence_head_attn[:n_words, :n_words] /= sentence_head_attn[:n_words, :n_words].sum(axis=-1)[..., None]
                        flat_sentence_head_attn = sentence_head_attn[tril_x, tril_y].tolist()  # n_words * (n_words + 1) / 2
                        flatten_model_attn[head].extend(flat_sentence_head_attn)

                # flatten human data
                flat_sentence_sub_data = sentence_sub_data[tril_x, tril_y].tolist()  # n_words * (n_words + 1) / 2
                flatten_subject_data.extend(flat_sentence_sub_data)

        # do regression for this subject
        X = np.array(flatten_model_attn).T
        y = np.array(flatten_subject_data)

        # no holdout
        reg_sub = LinearRegression().fit(X, y)
        score = reg_sub.score(X, y)

        # # do holdout, split train and test sets for LR
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        # reg_sub = LinearRegression().fit(X_train, y_train)
        # score = reg_sub.score(X_test, y_test)

        layer_r2_scores.append(score)

    # done regression for all subjects in this layer
    all_r2_scores.append(layer_r2_scores)

scores_array = np.array(all_r2_scores)
np.save(f'../../results/lr_scores_{dataset}/heads_vs_saccade/{attn_method + "_" if attn_method else ""}{file_prefix}{instr_prefix}{view}_{p}_{model_name}_{model_size}.npy', scores_array)
print(np.max(np.mean(scores_array, axis=1)))













