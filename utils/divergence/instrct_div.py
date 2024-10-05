import numpy as np
import pandas as pd
from scipy.stats import entropy as kl_div
from scipy.special import rel_entr
from sklearn.metrics.pairwise import cosine_similarity
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
instruction = 'ctrl_'
model_size = '13B'
method = 'cos'
attn_method = ''

# residue = eval(sys.argv[1])
# model_size = sys.argv[2]
# method = 'js'

d_size_layer = {'7B': 32, '13B': 40}
d_size_head = {'7B': 32, '13B': 40}
n_layers = d_size_layer[model_size]
n_heads = d_size_head[model_size]
div = kl_div if method == 'kl' else js_div

np.random.seed(42)
file_prefix = instruction

with open('../../sentence_data/reading_brain/sentences_p1.txt', 'r') as f:
    articles = f.read().strip().split('\n\n')
    article_sentences = [a.strip().split('\n') for a in articles]  # list(article) of list(sentence)
    sent_in_art = [len(a) for a in article_sentences]  # number of sentences in each article
    word_in_art_sent = [[len(s.strip().split()) for s in a] for a in
                        article_sentences]  # number of words in each article/sentence

with pd.ExcelWriter(f'../../results/divergence/instruct/{model_size}/{attn_method + "_" if attn_method else ""}{file_prefix}{method}.xlsx') as writer:
    # compare model attentions layer by layer
    for layer in range(n_layers):
        df_div = pd.DataFrame(
            columns=['llama', 'alpaca', 'vicuna', 'noise'])  # divergence in each sentence
        sheet_name = f'layer {layer}'

        # read attention of the three models: llama, alpaca, vicuna, and llama-hf (for noise ceiling)
        # shape: (n_arti, max_n_sents, n_head, max_sent_len, max_sent_len)
        llama_ori_attn = np.load(
            f'../../model_attention/reading_brain/llama/{model_size}/{attn_method if attn_method else "p1"}/rb_p1_layer{layer}.npy')
        llama_ins_attn = np.load(
            f'../../model_attention/reading_brain/llama/{model_size}/{attn_method if attn_method else "p1"}/{file_prefix}rb_p1_layer{layer}.npy')

        alpaca_ori_attn = np.load(
            f'../../model_attention/reading_brain/alpaca/{model_size}/{attn_method if attn_method else "p1"}/rb_p1_layer{layer}.npy')
        alpaca_ins_attn = np.load(
            f'../../model_attention/reading_brain/alpaca/{model_size}/{attn_method if attn_method else "p1"}/{file_prefix}rb_p1_layer{layer}.npy')

        vicuna_ori_attn = np.load(
            f'../../model_attention/reading_brain/vicuna/{model_size}/{attn_method if attn_method else "p1"}/rb_p1_layer{layer}.npy')
        vicuna_ins_attn = np.load(
            f'../../model_attention/reading_brain/vicuna/{model_size}/{attn_method if attn_method else "p1"}/{file_prefix}rb_p1_layer{layer}.npy')

        # for calculating noise
        vicuna_13b_attn = vicuna_ori_attn if model_size == '13B' else np.load(
            f'../../model_attention/reading_brain/vicuna/13B/{attn_method if attn_method else "p1"}/rb_p1_layer{layer}.npy')
        vicuna_old_attn = np.load(
            f'../../model_attention/reading_brain/vicuna-old/13B/{attn_method if attn_method else "p1"}/rb_p1_layer{layer}.npy')

        # calculate KL divergence in each article/sentence
        for arti in range(5):
            n_sent = sent_in_art[arti]
            for sentj in range(n_sent):
                n_words = word_in_art_sent[arti][sentj]
                # take the model attentions of this sentence
                # shape: (n_head, n_words, n_words)
                sent_llama_ori_attn = llama_ori_attn[arti, sentj, :, :n_words, :n_words]
                sent_llama_ins_attn = llama_ins_attn[arti, sentj, :, :n_words, :n_words]

                sent_alpaca_ori_attn = alpaca_ori_attn[arti, sentj, :, :n_words, :n_words]
                sent_alpaca_ins_attn = alpaca_ins_attn[arti, sentj, :, :n_words, :n_words]

                sent_vicuna_ori_attn = vicuna_ori_attn[arti, sentj, :, :n_words, :n_words]
                sent_vicuna_ins_attn = vicuna_ins_attn[arti, sentj, :, :n_words, :n_words]

                # for calculation noise
                sent_vicuna_13b_attn = vicuna_13b_attn[arti, sentj, :, :n_words, :n_words]
                sent_vicuna_old_attn = vicuna_old_attn[arti, sentj, :, :n_words, :n_words]

                for i, attn in enumerate([
                    sent_llama_ori_attn, sent_llama_ins_attn,
                    sent_alpaca_ori_attn, sent_alpaca_ins_attn,
                    sent_vicuna_ori_attn, sent_vicuna_ins_attn,
                    sent_vicuna_13b_attn, sent_vicuna_old_attn
                ]):
                    a = attn.sum(axis=-1)  # (n_head, max_sent_len('from' words))
                    assert not (a == 0).any(), \
                        f'layer {layer}, art {arti}, sent {sentj}, attn {i}\n{a}\n{np.where(a == 0)}'

                # divergence of residue attention needs min-max scaling, because there are negative values in it
                # if residue:
                #     sent_llama_ori_attn = min_max_preprocess(sent_llama_ori_attn)
                #     sent_alpaca_ori_attn = min_max_preprocess(sent_alpaca_ori_attn)
                #     sent_vicuna_attn = min_max_preprocess(sent_vicuna_attn)
                #
                #     sent_vicuna_13b_attn = min_max_preprocess(sent_vicuna_13b_attn)
                #     sent_vicuna_old_attn = min_max_preprocess(sent_vicuna_old_attn)

                # calculate layer-head-wise divergence with attn heads retained
                # shape of div result: n_head
                # divergence for each line ('to' words, sum <= 1), then mean over the 'from' words
                d0 = div(sent_llama_ori_attn, sent_llama_ins_attn, axis=-1).mean(axis=1)
                d1 = div(sent_alpaca_ori_attn, sent_alpaca_ins_attn, axis=-1).mean(axis=1)
                d2 = div(sent_vicuna_ori_attn, sent_vicuna_ins_attn, axis=-1).mean(axis=1)
                dn = div(sent_vicuna_13b_attn, sent_vicuna_old_attn, axis=-1).mean(axis=1)

                # layer-wise divergence with attn heads averaged
                # shape: scalar
                ah_d0 = d0.mean()  # ah stands for average head
                ah_d1 = d1.mean()
                ah_d2 = d2.mean()
                ah_dn = dn.mean()

                # append to dataframe in this layer
                row = pd.Series({
                    'llama': ah_d0,
                    'alpaca': ah_d1,
                    'vicuna': ah_d2,
                    'noise': ah_dn,
                })
                df_div = pd.concat(
                    [df_div, row.to_frame().T],
                    ignore_index=True)
            # no extra operation with articles
        # after finishing all the sentences for this layer, write to excel in one sheet
        df_div.to_excel(writer, sheet_name=sheet_name, index=False)
