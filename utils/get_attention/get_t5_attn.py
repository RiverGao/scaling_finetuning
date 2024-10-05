import torch
from transformers import T5Tokenizer, T5Model
import numpy as np
import pandas as pd
from torch.nn import functional as F

model_name = 't5-base'
part = 'cross'
name_size_map = {'t5-base': 'base', 't5-large': 'large'}
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
with open('sample_data/sample-sent.txt', 'r') as f:
    sentences = f.read().strip().split('\n')
lengths = []
for sent in sentences:
    lengths.append(len(sent.strip().split()))

# used for padding
max_sent_len = max(lengths)  # max number of words in all sentences

# BERT initialization
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5Model.from_pretrained(model_name)
n_layer = model.config.num_layers
n_head = model.config.num_heads

# iterate over sentences
attention_per_sentence = [[] for j in range(n_layer)]  # n_layer * n_sent * (n_head, max_sent_len, max_sent_len)
for sentence in sentences:
    s_words = sentence.split()
    print(tokenizer.tokenize(sentence))
    s_tokens = [x.replace('▁', '') for x in tokenizer.tokenize(sentence)]
    s_groups = token_groups(s_words, s_tokens)
    print(s_groups)

    encoder_input_ids = tokenizer(sentence, return_tensors='pt').input_ids
    decoder_input_ids = tokenizer(sentence, return_tensors="pt").input_ids
    # decoder_input_ids = model._shift_right(decoder_input_ids)
    assert len(encoder_input_ids[0]) == len(decoder_input_ids[0]) == len(tokenizer.tokenize(sentence)) + 1

    outputs = model(input_ids=encoder_input_ids, decoder_input_ids=decoder_input_ids, output_attentions=True)
    encoder_attentions = outputs.encoder_attentions  # n_layer * shape (1, n_head, L, L)
    decoder_attentions = outputs.decoder_attentions
    cross_attentions = outputs.cross_attentions

    assert encoder_attentions[0].size()[0] == decoder_attentions[0].size()[0] \
           == cross_attentions[0].size()[0] == len(encoder_input_ids)
    # assert len(encoder_attentions) == n_layer, \
    #     f'len of encoder attention: {len(encoder_attentions)}, n_layer: {n_layer}'
    # assert len(decoder_attentions) == n_layer, \
    #     f'len of decoder attention: {len(decoder_attentions)}, n_layer: {n_layer}'
    # assert len(cross_attentions) == n_layer, \
    #     f'len of cross attention: {len(cross_attentions)}, n_layer: {n_layer}'
    for lyr in range(n_layer):
        attn_tensor = None
        if part == 'encoder':
            attn_tensor = encoder_attentions[lyr].detach()
        elif part == 'decoder':
            attn_tensor = decoder_attentions[lyr].detach()
        elif part == 'cross':
            attn_tensor = cross_attentions[lyr].detach()
        else:
            raise ValueError(f'Unknown model part: {part}')
        # pad attention matrix with zeros
        assert attn_tensor.size()[1] == n_head
        attn_tensor = attn_tensor.squeeze()  # (n_head, L + 1, L + 1) because of </s> at the end

        attn_array = attn_tensor.numpy().mean(axis=0)[:-1, :-1]  # (L, L)
        merged_attn_array = merge_attentions(attn_array, s_groups)  # merge subword attention
        attn_tensor = torch.tensor(merged_attn_array)  # (L, L)
        pad_len = max_sent_len - attn_tensor.size()[-1]

        attn_tensor = F.pad(attn_tensor, (0, pad_len, 0, pad_len))  # (max_sent_len, max_sent_len)
        attention_per_sentence[lyr].append(attn_tensor)

# stack attention matrices of all sentences in an article
for lyr in range(n_layer):
    tensors_to_stack = attention_per_sentence[lyr]  # n_sent * (max_sent_len, max_sent_len)
    stacked_tensors = torch.stack(tensors_to_stack)  # (n_sent, max_sent_len, max_sent_len)
    np.save(f'attention/t5/{model_size}/{part}/attention_amr_layer{lyr}.npy', stacked_tensors)

