import torch
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
import numpy as np
import pandas as pd
from torch.nn import functional as F

model_name = 'gpt2-large'
name_size_map = {'gpt2': 'base', 'gpt2-large': 'large', 'gpt2-xl': 'xl', 'sberbank-ai/mGPT': 'multi'}
model_size = name_size_map[model_name]


def token_groups(words, tokens):
    groups = []
    words_iter = iter(words)
    word = next(words_iter)
    text_buf = ''
    id_buf = []
    for i, token in enumerate(tokens):
        if text_buf != word:
            text_buf = text_buf + token
            id_buf.append(i)
        else:
            groups.append(id_buf.copy())
            text_buf = token
            id_buf = [i]
            word = next(words_iter)
    groups.append(id_buf.copy())
    return groups


def merge_attentions(attn_mat, tok_groups):
    arrays = []  # temp store
    for group in tok_groups:
        array = 0
        for i in group:
            array += attn_mat[:, i]
        arrays.append(array)
    mat1 = np.stack(arrays).T

    arrays = []
    for group in tok_groups:
        array = np.mean([mat1[i, :] for i in group], axis=0)
        arrays.append(array)
    mat2 = np.stack(arrays)
    return mat2


# read sentences
with open('../../sentence_data/amr3.0/sample_data/sample-sent.txt', 'r') as f:
    sentences = f.read().strip().split('\n')
lengths = []
for sent in sentences:
    lengths.append(len(sent.strip().split()))

# used for padding
max_sent_len = max(lengths)  # max number of words in all sentences

# GPT initialization
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# model = GPT2Model.from_pretrained(model_name) if model_name != 'sberbank-ai/mGPT' \
#     else GPT2LMHeadModel.from_pretrained(model_name)
model = GPT2Model.from_pretrained(model_name)
n_layer = model.config.n_layer
n_head = model.config.n_head

# iterate over sentences
attention_per_sentence = [[] for j in range(n_layer)]  # n_layer * n_sent * (n_head, max_sent_len, max_sent_len)
for sentence in sentences:
    s_words = sentence.split()
    s_tokens = [x.replace('Ġ', '') for x in tokenizer.tokenize(sentence)]
    # print(s_words)
    s_groups = token_groups(s_words, s_tokens)

    encoded_input = tokenizer(sentence, return_tensors='pt')
    attentions = model(**encoded_input, output_attentions=True).attentions  # n_layer * shape (1, n_head, L, L)
    assert len(attentions) == n_layer, f'len of output: {len(attentions)}, n_layer: {n_layer}'
    for lyr in range(n_layer):
        attn_tensor = attentions[lyr].detach()
        # pad attention matrix with zeros
        assert attn_tensor.size()[1] == n_head
        attn_tensor = attn_tensor.squeeze()  # (n_head, L, L)

        attn_array = attn_tensor.numpy().mean(axis=0)  # (L, L)
        merged_attn_array = merge_attentions(attn_array, s_groups)  # merge subword attention
        attn_tensor = torch.tensor(merged_attn_array)  # (L, L)
        pad_len = max_sent_len - attn_tensor.size()[-1]

        attn_tensor = F.pad(attn_tensor, (0, pad_len, 0, pad_len))  # (max_sent_len, max_sent_len)
        attention_per_sentence[lyr].append(attn_tensor)

# stack attention matrices of all sentences in an article
for lyr in range(n_layer):
    tensors_to_stack = attention_per_sentence[lyr]  # n_sent * (max_sent_len, max_sent_len)
    stacked_tensors = torch.stack(tensors_to_stack)  # (n_sent, max_sent_len, max_sent_len)
    np.save(f'../../model_attention/amr/gpt2/{model_size}/attention_amr_layer{lyr}.npy', stacked_tensors)

