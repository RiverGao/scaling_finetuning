import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
from torch.nn import functional as F
import pdb
import pickle
import sys


model_path = sys.argv[1]  # path to your model
model_name = model_path.split('/')[-1]  # gpt2, Llama-3, phi-2, gemma, Mistral

token_begin = "Ġ" if "gpt2" in model_name or "Llama-3" in model_name or "phi-2" in model_name else "▁"
has_bos = False if model_name in ["phi-2", "phi-3-small"] else True


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

def merge_loss(loss_lst, tok_groups):
    losses = []
    # breakpoint()
    for group in tok_groups:
        loss = 0
        for i in group:
            loss += loss_lst[i]
        losses.append(loss)
    return losses


def get_per_token_loss(model, input_ids, labels, attention_mask=None):
    # Get model outputs
    outputs = model(input_ids, attention_mask=attention_mask, return_dict=True)
    logits = outputs.logits  # Shape: [batch_size, sequence_length, vocab_size]
    
    # Calculate per-token loss
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    # Reshape logits and labels for loss calculation
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Calculate loss for each position
    per_token_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    
    # Reshape loss back to [batch_size, sequence_length-1]
    per_token_loss = per_token_loss.view(shift_labels.shape)
    
    # If using attention mask, zero out losses for padding tokens
    if attention_mask is not None:
        shift_attention_mask = attention_mask[..., 1:].contiguous()
        per_token_loss = per_token_loss * shift_attention_mask
    
    breakpoint()
    return per_token_loss


# read sentences
with open('[data-path]/reading_brain_sentences.txt', 'r') as f:
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

# model initialization
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map= "auto"
)
model.eval()

# tokenizer = GPT2Tokenizer.from_pretrained(model_path, add_bos_token=True)
# model = GPT2LMHeadModel.from_pretrained(model_path, device_map= "auto")


nlls = []
# iterate over articles
for article in article_sentences:
    # iterate over sentences
    for sentence in article:
        # # no chat template
        # s_words = sentence.split()
        # s_tokens = [x.replace(token_begin, '') for x in tokenizer.tokenize(sentence)]
        # s_groups = token_groups(s_words, s_tokens)
        # print(s_words)
        # print(tokenizer.tokenize(sentence))
        # print(s_groups)
        # encoded_input = tokenizer(sentence, return_tensors='pt', add_special_tokens=True)
        # print(encoded_input)
        
        # with chat template
        messages = [{"role": "user", "content": sentence}]
        chat_sentence = tokenizer.apply_chat_template(messages, tokenize=False)
        s_words = chat_sentence.split(" ")
        s_tokens = [x.replace(token_begin, '') for x in tokenizer.tokenize(chat_sentence)]
        s_groups = token_groups(s_words, s_tokens)
        print(s_words)
        print(tokenizer.tokenize(chat_sentence))
        print(s_groups)
        encoded_input = tokenizer(chat_sentence, return_tensors='pt', add_special_tokens=False)
        print(encoded_input)
        
        # calculate loss
        encoded_input.to('cuda')
        input_ids = encoded_input.input_ids
        target_ids = input_ids.clone()
        target_len = len(target_ids[0])

        with torch.no_grad():
            nll_target = get_per_token_loss(
                model,
                input_ids,
                target_ids  
            )  # no attn_mask because we do not use padding
            
            print(nll_target)
            
            token_nll = (nll_target.cpu().tolist()[0])
            word_nll = merge_loss(token_nll, s_groups)
            nlls.append(word_nll)
            breakpoint()

# print(np.mean(nlls))
with open(f'[output-path]/{model_name}.pkl', 'wb') as file:
    pickle.dump(nlls, file)
