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
    # input and output shape: (n_words, n_words)
    _n_words, _ = sent_attn.shape
    ret = np.zeros_like(sent_attn)
    flat_idx = tril_idx(_n_words)

    flat_sent_attn = sent_attn[flat_idx]  # 1-D array of size: n_words * (n_words + 1) // 2
    min_attn = np.min(flat_sent_attn)
    max_attn = np.max(flat_sent_attn)
    flat_sent_attn = (flat_sent_attn - min_attn) * (max_attn - min_attn) + 1e-5

    apply_values(flat_sent_attn, ret, _n_words)

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
    p = p[1:, :-1]
    q = q - np.triu(q)
    q = q[1:, :-1]

    p = p / np.sum(p, axis=axis, keepdims=True)
    q = q / np.sum(q, axis=axis, keepdims=True)

    assert p.shape == q.shape
    flat_idx = tril_idx(p.shape[1])

    p_flat = p[flat_idx]  # (n_tril)
    q_flat = q[flat_idx]

    p_norm = np.linalg.norm(p_flat, 2)  # scalar
    q_norm = np.linalg.norm(q_flat, 2)

    p = np.log(p_flat * p_norm ** 2).reshape(1,-1)
    q = np.log(q_flat * q_norm ** 2).reshape(1,-1)

    # sim = p @ q / (np.linalg.norm(p, 2) * np.linalg.norm(q, 2))
    sim = cosine_similarity(p, q)
    # sim = sim.reshape((len(sim), 1))
    return sim


method = 'cos'
residue = False
instruction = ''
attn_method = ''
# residue = eval(sys.argv[1])

d_size_layer = {'7B': 32, '13B': 40}
d_size_head = {'7B': 32, '13B': 40}
d_div = {'kl': kl_div, 'js': js_div, 'cos': cos_sim}

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

with pd.ExcelWriter(f'../../results/divergence/size/{attn_method + "_" if attn_method else ""}{file_prefix}{method}-size.xlsx') as writer:
    # compare model attentions part by part, each part is 1/4 of the layers
    for part in range(4):
        sheet_name = f'part {part}'
        df_div = pd.DataFrame(columns=['llama', 'alpaca', 'vicuna', 'noise'])  # divergence in each sentence
        # df_div = pd.DataFrame(columns=['llama', 'alpaca', 'vicuna'])

        # for the 7B model
        llama7B_attn_in_part = []
        alpaca7B_attn_in_part = []
        vicuna7B_attn_in_part = []
        vicuna7B_old_attn_in_part = []
        for layer in range(part * 8, (part + 1) * 8):
            # read attention of the three models: llama, alpaca, vicuna
            # shape: (n_arti, max_n_sents, n_head, max_sent_len, max_sent_len)
            llama7B_attn_in_part.append(np.load(f'../../model_attention/reading_brain/llama/7B/{attn_method if attn_method else "p1"}/{file_prefix}rb_p1_layer{layer}.npy'))

            alpaca7B_attn_in_part.append(np.load(f'../../model_attention/reading_brain/alpaca/7B/{attn_method if attn_method else "p1"}/{file_prefix}rb_p1_layer{layer}.npy'))

            vicuna7B_attn_in_part.append(np.load(f'../../model_attention/reading_brain/vicuna/7B/{attn_method if attn_method else "p1"}/{file_prefix}rb_p1_layer{layer}.npy'))
            vicuna7B_old_attn_in_part.append(np.load(f'../../model_attention/reading_brain/vicuna-old/7B/{attn_method if attn_method else "p1"}/{file_prefix}rb_p1_layer{layer}.npy'))

        assert len(llama7B_attn_in_part) == 8
        llama7B_attn = np.mean(llama7B_attn_in_part, axis=0).mean(axis=2)
        alpaca7B_attn = np.mean(alpaca7B_attn_in_part, axis=0).mean(axis=2)
        vicuna7B_attn = np.mean(vicuna7B_attn_in_part, axis=0).mean(axis=2)
        vicuna7B_old_attn = np.mean(vicuna7B_old_attn_in_part, axis=0).mean(axis=2)
        # shape: (n_arti, max_n_sents, max_sent_len, max_sent_len), averaged over heads

        # for the 13B model
        llama13B_attn_in_part = []
        alpaca13B_attn_in_part = []
        vicuna13B_attn_in_part = []
        vicuna13B_old_attn_in_part = []
        for layer in range(part * 10, (part + 1) * 10):
            # read attention of the three models: llama, alpaca, vicuna
            # shape: (n_arti, max_n_sents, n_head, max_sent_len, max_sent_len)
            llama13B_attn_in_part.append(
                np.load(f'../../model_attention/reading_brain/llama/13B/{attn_method if attn_method else "p1"}/{file_prefix}rb_p1_layer{layer}.npy'))
            alpaca13B_attn_in_part.append(
                np.load(f'../../model_attention/reading_brain/alpaca/13B/{attn_method if attn_method else "p1"}/{file_prefix}rb_p1_layer{layer}.npy'))
            vicuna13B_attn_in_part.append(
                np.load(f'../../model_attention/reading_brain/vicuna/13B/{attn_method if attn_method else "p1"}/{file_prefix}rb_p1_layer{layer}.npy'))
            vicuna13B_old_attn_in_part.append(
                np.load(f'../../model_attention/reading_brain/vicuna-old/13B/{attn_method if attn_method else "p1"}/{file_prefix}rb_p1_layer{layer}.npy'))
        assert len(llama13B_attn_in_part) == 10
        llama13B_attn = np.mean(llama13B_attn_in_part, axis=0).mean(axis=2)
        alpaca13B_attn = np.mean(alpaca13B_attn_in_part, axis=0).mean(axis=2)
        vicuna13B_attn = np.mean(vicuna13B_attn_in_part, axis=0).mean(axis=2)
        vicuna13B_old_attn = np.mean(vicuna13B_old_attn_in_part, axis=0).mean(axis=2)
        # must average over attention heads
        # shape: (n_arti, max_n_sents, max_sent_len, max_sent_len)

        # calculate KL divergence in each article/sentence
        for arti in range(5):
            n_sent = sent_in_art[arti]
            for sentj in range(n_sent):
                n_words = word_in_art_sent[arti][sentj]
                # take the model attentions of this sentence
                # shape: (n_words, n_words)
                sent_llama7B_attn = llama7B_attn[arti, sentj, :n_words, :n_words]
                sent_alpaca7B_attn = alpaca7B_attn[arti, sentj, :n_words, :n_words]
                sent_vicuna7B_attn = vicuna7B_attn[arti, sentj, :n_words, :n_words]
                sent_vicuna7B_old_attn = vicuna7B_old_attn[arti, sentj, :n_words, :n_words]

                sent_llama13B_attn = llama13B_attn[arti, sentj, :n_words, :n_words]
                sent_alpaca13B_attn = alpaca13B_attn[arti, sentj, :n_words, :n_words]
                sent_vicuna13B_attn = vicuna13B_attn[arti, sentj, :n_words, :n_words]
                sent_vicuna13B_old_attn = vicuna13B_old_attn[arti, sentj, :n_words, :n_words]

                for i, attn in enumerate(
                        [sent_llama7B_attn,
                         sent_alpaca7B_attn,
                         sent_vicuna7B_attn,
                         sent_vicuna7B_old_attn,
                         sent_llama13B_attn,
                         sent_alpaca13B_attn,
                         sent_vicuna13B_attn,
                         sent_vicuna13B_old_attn]):
                    a = attn.sum(axis=-1)  # (max_sent_len('from' words))
                    assert not (a == 0).any(), \
                        f'layer {layer}, art {arti}, sent {sentj}, attn {i}\n{a}\n{np.where(a==0)}'

                # divergence of residue attention needs min-max scaling, because there are negative values in it
                if residue:
                    sent_llama7B_attn = min_max_preprocess(sent_llama7B_attn)
                    sent_alpaca7B_attn = min_max_preprocess(sent_alpaca7B_attn)
                    sent_vicuna7B_attn = min_max_preprocess(sent_vicuna7B_attn)
                    sent_vicuna7B_old_attn = min_max_preprocess(sent_vicuna7B_old_attn)

                    sent_llama13B_attn = min_max_preprocess(sent_llama13B_attn)
                    sent_alpaca13B_attn = min_max_preprocess(sent_alpaca13B_attn)
                    sent_vicuna13B_attn = min_max_preprocess(sent_vicuna13B_attn)
                    sent_vicuna13B_old_attn = min_max_preprocess(sent_vicuna13B_old_attn)

                # calculate layer-wise divergence with attn heads already averaged
                # shape of div result: scalar
                # divergence for each line ('to' words, sum = 1), then mean over the 'from' words
                ahdl = div(sent_llama7B_attn, sent_llama13B_attn, axis=-1).mean()
                ahda = div(sent_alpaca7B_attn, sent_alpaca13B_attn, axis=-1).mean()
                ahdv = div(sent_vicuna7B_attn, sent_vicuna13B_attn, axis=-1).mean()

                ahdn1 = div(sent_vicuna7B_attn, sent_vicuna7B_old_attn, axis=-1).mean()
                ahdn2 = div(sent_vicuna13B_attn, sent_vicuna13B_old_attn, axis=-1).mean()
                # ahdn = (ahdn1 + ahdn2) / 2
                ahdn = ahdn2

                # append to dataframe in this layer
                row = pd.Series({'llama': ahdl, 'alpaca': ahda, 'vicuna': ahdv, 'noise': ahdn})
                # row = pd.Series({'llama': ahdl, 'alpaca': ahda, 'vicuna': ahdv})
                df_div = pd.concat(
                    [df_div, row.to_frame().T],
                    ignore_index=True)
            # no extra operation with articles
        # after finishing all the sentences for this layer, write to excel in one sheet
        df_div.to_excel(writer, sheet_name=sheet_name, index=False)
