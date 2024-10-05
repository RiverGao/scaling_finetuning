import numpy as np
import pickle
import itertools
from matplotlib import pyplot as plt

lbl_code = 5
d_lbl_task = [
    'attribute',  # 定语
    'adverbial',  # 状语
    'predicate',  # 主谓 + 谓宾
    'self',
    'prev',
    'start'
]
lbl_task = d_lbl_task[lbl_code]


def plot(words, labels, title):
    fig, ax = plt.subplots(figsize=(12, 12))
    saccade_graph = ax.matshow(labels)
    ax.set_xticks([i for i in range(len(words))], [w for w in words], rotation=30, ha='left')
    ax.set_yticks([i for i in range(len(words))], [w for w in words], rotation=60, ha='right')
    ax.set_xlabel('Word To')
    ax.set_ylabel('Word From')
    ax.set_title(title)
    fig.colorbar(saccade_graph, ax=ax)
    fig.savefig(f'../../results/figs/reading_brain/label/{lbl_task}-example.png', dpi=80)
    plt.close(fig)


def parse_index(idx):
    # idx format: "int", or "(int_start, int_end)"
    # return: list of true indices
    if '(' not in idx:
        return [eval(idx)]
    else:
        start, end = eval(idx)
        assert end > start
        return [i for i in range(start, end + 1)]


with open('../../sentence_data/reading_brain/sentences_p1.txt', 'r') as f_sen:
    articles = f_sen.read().strip().split('\n\n')

with open(f'label_{lbl_task}.pkl', 'rb') as f:
    label_data = pickle.load(f)

for ai, article in enumerate(articles):
    # for each article
    print('\n' + '*' * 13 + f'\n* Article {ai} *\n' + '*' * 13)
    sentences = article.split('\n')
    for si, sent in enumerate(sentences):
        # for each sentence
        print(f'Task {lbl_task}, Sentence {si}:\n{sent}')
        tokens = sent.split()
        n_token = len(tokens)
        lbl_mat = label_data[ai][si]

        # start interaction with user
        tokens_with_num = [f'{i}:{tok}' for i, tok in enumerate(tokens)]
        print('  '.join(tokens_with_num))
        print(np.array(lbl_mat))
        plot(tokens, np.array(lbl_mat), f'Article {ai}, Sentence {si}, {lbl_task} Label')
        print('Input format: <from>-<to>. When you finish, please enter "d".')
        while 1:
            # accept label input for one sentence
            try:
                p = input()
                if p == 'd':
                    print('=' * 80)
                    break

                if '-' not in p:
                    print('Invalid input format!')
                    raise NotImplementedError

                i_from, i_to = [parse_index(idx) for idx in p.split('-')]
                # index of the from-to tokens, two formats: int, or (int_start, int_end)

                if max(i_from) >= n_token or max(i_to) >= n_token:
                    print(f"Indices ({i_from}, {i_to}) are out of range {n_token}")
                    raise NotImplementedError

                for f, t in itertools.product(i_from, i_to):
                    # print(f, t)
                    lbl_mat[f][t] = 1

                # save the change and print the updated result
                with open(f'label_{lbl_task}.pkl', 'wb') as f:
                    pickle.dump(label_data, f)
                print(np.array(lbl_mat))

            except NotImplementedError:
                pass
        # finish labeling one sentence
    # finish labeling all sentences in one article
# after finishing all articles
print('Finished!')
