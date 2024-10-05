import numpy as np
import pandas as pd
from scipy.stats import entropy as kl_div
from scipy.special import rel_entr
from sklearn.metrics.pairwise import cosine_similarity
import sys


def tril_idx(n, include_diag=True):
    # 00, 01, 11, 02, 12, 22
    x = []
    y = []
    for i in range(n):
        for j in range(n):
            if include_diag and j > i:
                break
            elif (not include_diag) and j >= i:
                break
            x.append(i)
            y.append(j)
    return np.array(x), np.array(y)


def apply_values(values, target, size, include_diag=True):
    # size is the width (also height) of the original value matrix, i.e. n_words
    it = iter(values)
    for i in range(size):
        for j in range(size):
            if include_diag and j > i:
                break
            elif (not include_diag) and j >= i:
                break
            target[i, j] += next(it)


def min_max_preprocess(sent_attn):
    # input and output shape: (n_head, n_words, n_words)
    _n_head, _n_words, _ = sent_attn.shape
    ret = np.zeros_like(sent_attn)
    flat_idx = tril_idx(_n_words)

    for head in range(_n_head):
        head_attn = sent_attn[head, :, :]
        flat_head_attn = head_attn[flat_idx]  # 1-D array of size: n_words * (n_words + 1) // 2
        min_attn = np.min(flat_head_attn)
        max_attn = np.max(flat_head_attn)
        flat_head_attn = (flat_head_attn - min_attn) * (max_attn - min_attn) + 1e-5

        apply_values(flat_head_attn, ret[head], _n_words)

    return ret


def js_div(p, q, base=None, *, axis=0, keepdims=False):
    p = np.asarray(p)
    q = np.asarray(q)
    p = p / np.sum(p, axis=axis, keepdims=True)
    q = q / np.sum(q, axis=axis, keepdims=True)
    m = (p + q) / 2.0
    left = rel_entr(p, m)
    right = rel_entr(q, m)
    left_sum = np.sum(left, axis=axis, keepdims=keepdims)
    right_sum = np.sum(right, axis=axis, keepdims=keepdims)
    js = left_sum + right_sum
    if base is not None:
        js /= np.log(base)
    return js / 2.0


def cos_sim(p, q, axis=0):
    # p, q: (n_word, n_word)
    p = np.asarray(p)
    q = np.asarray(q)

    p = p - np.triu(p)
    p = p[:, 1:, :-1]
    q = q - np.triu(q)
    q = q[:, 1:, :-1]

    p = p / np.sum(p, axis=axis, keepdims=True)
    q = q / np.sum(q, axis=axis, keepdims=True)

    assert p.shape == q.shape
    flat_idx = tril_idx(p.shape[1])

    p_flat = np.array([a[flat_idx] for a in p])  # (32, n_tril)
    q_flat = np.array([a[flat_idx] for a in q])

    p_norm = np.linalg.norm(p_flat, 2, axis=1).reshape((-1, 1))  # (32, 1)
    q_norm = np.linalg.norm(q_flat, 2, axis=1).reshape((-1, 1))

    p = np.log(p_flat * p_norm ** 2)
    q = np.log(q_flat * q_norm ** 2)

    # sim = p @ q / (np.linalg.norm(p, 2) * np.linalg.norm(q, 2))
    sim = cosine_similarity(p, q).diagonal()
    sim = sim.reshape((len(sim), 1))
    return sim


residue = False
instruction = ''
model_size = '13B'
method = 'cos'
attn_method = ''

# residue = eval(sys.argv[1])
# model_size = sys.argv[2]
# method = 'js'

d_size_layer = {'7B': 32, '13B': 40}
d_size_head = {'7B': 32, '13B': 40}
d_div = {'kl': kl_div, 'js': js_div, 'cos': cos_sim}

n_layers = d_size_layer[model_size]
n_heads = d_size_head[model_size]
div = d_div[method]

np.random.seed(42)
file_prefix = 'residue_' if residue else ''
file_prefix = instruction + file_prefix

with open('../../sentence_data/reading_brain/sentences_p1.txt', 'r') as f:
    articles = f.read().strip().split('\n\n')
    article_sentences = [a.strip().split('\n') for a in articles]  # list(article) of list(sentence)
    sent_in_art = [len(a) for a in article_sentences]  # number of sentences in each article
    word_in_art_sent = [[len(s.strip().split()) for s in a] for a in
                        article_sentences]  # number of words in each article/sentence

with pd.ExcelWriter(
        f'../../results/divergence/name/{model_size}/{attn_method + "_" if attn_method else ""}{file_prefix}{method}-name.xlsx') as writer:
    # compare model attentions layer by layer
    for layer in range(n_layers):
        df_div = pd.DataFrame(
            columns=['llama-alpaca', 'llama-vicuna', 'alpaca-vicuna', 'noise'])  # divergence in each sentence
        sheet_name = f'layer {layer}'

        # read attention of the three models: llama, alpaca, vicuna, and llama-hf (for noise ceiling)
        # shape: (n_arti, max_n_sents, n_head, max_sent_len, max_sent_len)
        llama_attn = np.load(
            f'../../model_attention/reading_brain/llama/{model_size}/{attn_method if attn_method else "p1"}/{file_prefix}rb_p1_layer{layer}.npy')
        alpaca_attn = np.load(
            f'../../model_attention/reading_brain/alpaca/{model_size}/{attn_method if attn_method else "p1"}/{file_prefix}rb_p1_layer{layer}.npy')
        vicuna_attn = np.load(
            f'../../model_attention/reading_brain/vicuna/{model_size}/{attn_method if attn_method else "p1"}/{file_prefix}rb_p1_layer{layer}.npy')
        # for calculating noise
        vicuna_13b_attn = vicuna_attn if model_size == '13B' else np.load(
            f'../../model_attention/reading_brain/vicuna/13B/{attn_method if attn_method else "p1"}/{file_prefix}rb_p1_layer{layer}.npy')
        vicuna_old_attn = np.load(
            f'../../model_attention/reading_brain/vicuna-old/13B/{attn_method if attn_method else "p1"}/{file_prefix}rb_p1_layer{layer}.npy')

        # calculate KL divergence in each article/sentence
        for arti in range(5):
            n_sent = sent_in_art[arti]
            for sentj in range(n_sent):
                n_words = word_in_art_sent[arti][sentj]
                # take the model attentions of this sentence
                # shape: (n_head, n_words, n_words)
                sent_llama_attn = llama_attn[arti, sentj, :, :n_words, :n_words]
                sent_alpaca_attn = alpaca_attn[arti, sentj, :, :n_words, :n_words]
                sent_vicuna_attn = vicuna_attn[arti, sentj, :, :n_words, :n_words]

                # for calculation noise
                sent_vicuna_13b_attn = vicuna_13b_attn[arti, sentj, :, :n_words, :n_words]
                sent_vicuna_old_attn = vicuna_old_attn[arti, sentj, :, :n_words, :n_words]

                for i, attn in enumerate([sent_llama_attn, sent_alpaca_attn, sent_vicuna_attn, sent_vicuna_old_attn]):
                    # for i, attn in enumerate([sent_llama_attn, sent_alpaca_attn, sent_vicuna_attn]):
                    a = attn.sum(axis=-1)  # (n_head, max_sent_len('from' words))
                    assert not (a == 0).any(), \
                        f'layer {layer}, art {arti}, sent {sentj}, attn {i}\n{a}\n{np.where(a == 0)}'

                # divergence of residue attention needs min-max scaling, because there are negative values in it
                if residue:
                    sent_llama_attn = min_max_preprocess(sent_llama_attn)
                    sent_alpaca_attn = min_max_preprocess(sent_alpaca_attn)
                    sent_vicuna_attn = min_max_preprocess(sent_vicuna_attn)

                    sent_vicuna_13b_attn = min_max_preprocess(sent_vicuna_13b_attn)
                    sent_vicuna_old_attn = min_max_preprocess(sent_vicuna_old_attn)

                # calculate layer-head-wise divergence with attn heads retained
                # shape of div result: n_head
                # divergence for each line ('to' words, sum <= 1), then mean over the 'from' words
                d01 = div(sent_llama_attn, sent_alpaca_attn, axis=-1).mean(axis=1)
                d02 = div(sent_llama_attn, sent_vicuna_attn, axis=-1).mean(axis=1)
                d12 = div(sent_alpaca_attn, sent_vicuna_attn, axis=-1).mean(axis=1)
                d22 = div(sent_vicuna_13b_attn, sent_vicuna_old_attn, axis=-1).mean(axis=1)

                # d10 = div(sent_alpaca_attn, sent_llama_attn, axis=-1).mean(axis=1)
                # d20 = div(sent_vicuna_attn, sent_llama_attn, axis=-1).mean(axis=1)
                # d21 = div(sent_vicuna_attn, sent_alpaca_attn, axis=-1).mean(axis=1)

                # layer-wise divergence with attn heads averaged
                # shape: scalar
                ah_d01 = d01.mean()  # ah stands for average head
                ah_d02 = d02.mean()
                ah_d12 = d12.mean()
                ah_d22 = d22.mean()

                # ah_d10 = d10.mean()  # ah stands for average head
                # ah_d20 = d20.mean()
                # ah_d21 = d21.mean()

                # append to dataframe in this layer
                row = pd.Series({
                    'llama-alpaca': ah_d01,
                    'llama-vicuna': ah_d02,
                    'alpaca-vicuna': ah_d12,
                    'noise': ah_d22,
                })
                df_div = pd.concat(
                    [df_div, row.to_frame().T],
                    ignore_index=True)
            # no extra operation with articles
        # after finishing all the sentences for this layer, write to excel in one sheet
        df_div.to_excel(writer, sheet_name=sheet_name, index=False)
