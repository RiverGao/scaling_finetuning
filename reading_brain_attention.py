import torch
from transformers import LlamaTokenizer, LlamaModel
import numpy as np
from torch.nn import functional as F
import pdb
import sys

model_size = sys.argv[1]
model_path = f'/home/gaocj/llama-test/hf-ckpt/{model_size}'
# model_path = 'decapoda-research/llama-13b-hf'
p = eval(sys.argv[2])
has_instruction = True

# instruction = 'Please translate sentence into German:'
# prefix = 'instr_' if has_instruction else ''

# instruction = 'Please paraphrase this sentence:'
# prefix = 'para_' if has_instruction else ''

instruction = 'Cigarette first steel convenience champion.'
prefix = 'ctrl_' if has_instruction else ''


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
with open(f'text/reading_brain/sentences_p{p}.txt', 'r') as f:
    articles = f.read().strip().split('\n\n')

article_sentences = []  # List[List[str]]
n_sents = []  # number of sentences in each article
n_words = []  # number of words in each sentence
for article in articles:
    sentences = article.strip().split('\n')
    article_sentences.append(sentences)
    n_sents.append(len(sentences))

    for sentence in sentences:
        n_words.append(len(sentence.strip().split()))  # number of words in this sentence

# used for padding
max_n_sents = max(n_sents)  # max number of sentences in all articles
max_sent_len = max(n_words)  # max number of words in all sentences

# model initialization
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaModel.from_pretrained(
    model_path, 
    device_map= "auto", 
    load_in_8bit=False)
# model.to('cuda')
model.eval()

n_layer = model.config.num_hidden_layers
n_head = model.config.num_attention_heads

# tokenize the instruction
if has_instruction:
    tok_instruct = tokenizer.tokenize(instruction)
    print(f"Tokenized instruction: {' '.join(tok_instruct)}")
    len_instruct = len(tok_instruct)

# iterate over articles
tensor_per_article = [[] for j in range(n_layer)]  # n_layer * n_arti * (max_n_sents, n_head, max_sent_len, max_sent_len)
for article in article_sentences:
    # iterate over sentences
    attention_per_sentence = [[] for j in range(n_layer)]  # n_layer * n_sent * (n_head, max_sent_len, max_sent_len)
    for sentence in article:
        s_words = sentence.split()
        print(tokenizer.tokenize(sentence))
        # assert 0
        s_tokens = [x.replace('▁', '') for x in tokenizer.tokenize(sentence)]
        print(s_words)
        s_groups = token_groups(s_words, s_tokens)
        print(s_groups)

        if not has_instruction:
            encoded_input = tokenizer(sentence, return_tensors='pt')
        else:
            encoded_input = tokenizer(' '.join([instruction, sentence]), return_tensors='pt')
        
        encoded_input.to('cuda')
        attentions = model(**encoded_input, output_attentions=True).attentions  # n_layer * shape (1, n_head, L, L)
        assert len(attentions) == n_layer, f'len of output: {len(attentions)}, n_layer: {n_layer}'
        for lyr in range(n_layer):
            print(f'Layer {lyr}:')
            attn_tensor = attentions[lyr].detach().cpu()
            assert not np.isnan(np.sum(attn_tensor.numpy())), f"layer {lyr} has NaN attentions"
            
            # pad attention matrix with zeros
            assert attn_tensor.size()[1] == n_head
            attn_tensor = attn_tensor.squeeze()  # (n_head, L + 1, L + 1), because of [BOS]
            print(attn_tensor.size())

            if not has_instruction:
                attn_array = attn_tensor.numpy()[:, 1:, 1:]  # do not average attention heads
            else:
                attn_array = attn_tensor.numpy()[:, len_instruct + 1:, len_instruct + 1:]

            print(attn_array.shape)
            list_head_attn = []
            for head in range(n_head):
                head_attn_array = attn_array[head]
                merged_head_attn_array = merge_attentions(head_attn_array, s_groups)  # merge subword attention
                list_head_attn.append(merged_head_attn_array)
            merged_attn_array = np.stack(list_head_attn)
            print(merged_attn_array.shape, len(s_words))
            attn_tensor = torch.tensor(merged_attn_array)  # (n_head, L, L)
            pad_len = max_sent_len - attn_tensor.size()[-1]

            attn_tensor = F.pad(attn_tensor, (0, pad_len, 0, pad_len))  # (n_head, max_sent_len, max_sent_len)
            print(attn_tensor.size())
            attention_per_sentence[lyr].append(attn_tensor)

    # stack attention matrices of all sentences in an article
    for lyr in range(n_layer):
        tensors_to_stack = attention_per_sentence[lyr]  # n_sent * (n_head, max_sent_len, max_sent_len)
        stacked_tensors = torch.stack(tensors_to_stack)  # (n_sent, n_head, max_sent_len, max_sent_len)
        pad_n = max_n_sents - len(tensors_to_stack)  # padding for article sentences
        stacked_tensors = F.pad(stacked_tensors,
                                (0, 0, 0, 0, 0, 0, 0, pad_n))  # (max_n_sents, n_head, max_sent_len, max_sent_len)
        print(f'Layer {lyr} article shape: {stacked_tensors.size()}')
        tensor_per_article[lyr].append(stacked_tensors)

# stack tensors of all articles
for lyr in range(n_layer):
    tensors_to_stack = tensor_per_article[lyr]
    stacked_tensors = torch.stack(tensors_to_stack)  # (n_arti, max_n_sents, n_head, max_sent_len, max_sent_len)
    stacked_np = stacked_tensors.numpy()
    print(f'All final shape: {stacked_np.shape}')
    np.save(f'attentions/reading_brain/{model_size}/p{p}/{prefix}rb_p{p}_layer{lyr}.npy', stacked_np)
