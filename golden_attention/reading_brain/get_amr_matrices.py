from penman import decode, surface
import pickle
import itertools
import numpy as np
from transition_amr_parser.parse import AMRParser


def connect(src_idx, tgt_idx):
    return_tuples = []
    for si in src_idx:
        for ti in tgt_idx:
            return_tuples.append((si, ti))
    return return_tuples


class IndexedSentence:
    def __init__(self, text: str):
        self.text = text
        self.words = text.strip().split()
        self.index = self._build_index()

    def _build_index(self):
        index_list = []
        word_counter = 0
        for ic, c in enumerate(self.text):
            index_list.append(word_counter)
            if c == ' ':
                word_counter += 1
        return index_list

    def search(self, b: int, e: int):
        word_indices = set(self.index[b: e])
        assert len(word_indices) == 1, word_indices
        return word_indices.pop()


with open('../../sentence_data/reading_brain/sentences_p1.txt', 'r') as f_sen:
    articles = f_sen.read().strip().split('\n\n')
parser = AMRParser.from_pretrained('AMR3-structbart-L')
output_matrices = []  # n_article * n_sentence * n_token * n_token

for article in articles:
    sentences = article.strip().split('\n')
    article_matrices = []  # n_sentence * n_token * n_token
    for s in sentences:
        # for each sentence
        tokens_sep_punc, positions = parser.tokenize(s)  # punctuations are separated from words
        tokens_with_punc = s.split()  # punctuations are concatenated with words

        indexed_sent = IndexedSentence(s)  # index chars with words

        annotations, machines = parser.parse_sentence(tokens_sep_punc)
        print(annotations)
        graph = decode(annotations)
        triples = graph.triples  # [('a', ':instance', 'and'), ('a', ':op1', 'i'), ... ]
        edges = graph.edges()  # [Edge(source='a', role=':op1', target='i'), ... ]
        align = surface.alignments(graph)  # {('i', ':instance', 'international'): Alignment((0,), prefix='e.'), ... }
        variables = graph.variables()  # {'g', 'a', 't', ... }

        # first construct the alignment dict
        dict_align = dict()  # {'i': [0], 'g': [2], ... }, all keys in this dict are concrete nodes
        # the word indices in dict_align is based on tokens_sep_punc, and needs to be reindexed
        for al in align.items():
            node = al[0][0]  # the first element of the key, i.e. name of the node
            idx = al[1].indices[0]  # the first element of the value, i.e. the alignment index
            if node in dict_align:
                dict_align[node].append(idx)
            else:
                dict_align[node] = [idx]

        # print(dict_align)
        # print(s)
        # print(triples)
        # print(edges)
        # print(variables)

        # convert alignments of :name children to their parents
        for e in edges:
            src = e.source
            tgt = e.target
            role = e.role
            assert src != tgt

            if role == ':name' and tgt in dict_align:
                # print(src, tgt)
                # src: person, location, etc
                # tgt: the real name of them
                dict_align[src] = dict_align[tgt]
                dict_align.pop(tgt)

        # print(dict_align)

        # then describe all the connections in the AMR graph with an n_token * n_token list[list]
        n_token = len(tokens_sep_punc)
        n_word = len(tokens_with_punc)
        connection_mat = [[0] * n_word for i in range(n_word)]  # initialize with all zero
        dict_children = dict().fromkeys(variables)  # all nodes (concrete & abstract) to their children list
        for e in edges:
            src = e.source
            tgt = e.target
            assert src != tgt  # assume no self-edge

            # add the target to the source's children list
            if not dict_children[src]:
                dict_children[src] = [tgt]
            else:
                if tgt not in dict_children[src]:
                    dict_children[src].append(tgt)

            # skip direct connections of abstract nodes
            if not (src in dict_align and tgt in dict_align):
                continue

            # add direct connections to the matrix
            idx_source = dict_align[src]  # list of index of tokens
            idx_target = dict_align[tgt]  # list of index of tokens
            tuple_edges = connect(idx_source, idx_target)
            for te in tuple_edges:
                tok_from, tok_to = te  # based on tokens_sep_punc
                begin_from, end_from = positions[tok_from]  # beginning and end index of the from-word
                begin_to, end_to = positions[tok_to]  # beginning and end index of the to-word

                word_from = indexed_sent.search(begin_from, end_from)
                word_to = indexed_sent.search(begin_to, end_to)

                connection_mat[word_from][word_to] = 1  # only consider occurrence
                connection_mat[word_to][word_from] = 1  # no direction considered

        print(np.array(connection_mat))
        print('='*80)

        article_matrices.append(connection_mat)
    output_matrices.append(article_matrices)

with open('label_amr.pkl', 'wb') as f:
    pickle.dump(output_matrices, f)

