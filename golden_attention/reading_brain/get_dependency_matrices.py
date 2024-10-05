import pickle
import stanza
import numpy as np

with open('../../sentence_data/reading_brain/sentences_p1.txt', 'r') as f_sen:
    articles = f_sen.read().strip().split('\n\n')
nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse')
output_matrices = []  # n_article * n_sentence * n_token * n_token


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


for article in articles:
    sentences = article.strip().split('\n')
    article_matrices = []  # n_sentence * n_token * n_token
    for sent in sentences:
        words = sent.strip().split()
        n_words = len(words)
        dependency_mat = [[0] * n_words for i in range(n_words)]
        print(sent)

        indexed_sent = IndexedSentence(sent)
        doc = nlp(sent)
        deps = doc.sentences[0].dependencies
        for d in deps:
            from_info, relation, to_info = d
            assert to_info.text != 'ROOT'
            if from_info.text == 'ROOT':
                continue

            i_from = indexed_sent.search(from_info.start_char, from_info.end_char)
            i_to = indexed_sent.search(to_info.start_char, to_info.end_char)
            # assert from_info.text == words[i_from], f'original word: {words[i_from]}, parsed word: {from_info.text}'
            # assert to_info.text == words[i_to], f'original word: {words[i_to]}, parsed word: {to_info.text}'
            dependency_mat[i_from][i_to] = 1
            dependency_mat[i_to][i_from] = 1

        print(np.array(dependency_mat))
        article_matrices.append(dependency_mat)
    output_matrices.append(article_matrices)

with open('label_dependency.pkl', 'wb') as f:
    pickle.dump(output_matrices, f)

