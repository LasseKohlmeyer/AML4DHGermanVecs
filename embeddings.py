from typing import List

from gensim import models as gensim
from gensim.test.utils import get_tmpfile
from gensim.models import Phrases, Word2Vec
import numpy as np


class Embeddings:
    @staticmethod
    def calculate_vectors(tokens=List[str], use_phrases: bool = False,
                          w2v_model=gensim.Word2Vec, dim: int = 300, window=10) -> gensim.KeyedVectors:
        seed = 42
        if use_phrases:
            bigram_transformer = Phrases(tokens)
            model = w2v_model(bigram_transformer[tokens], size=dim, window=window, min_count=1, workers=4, seed=seed)
        else:
            model = Word2Vec(tokens, size=dim, window=window, min_count=1, workers=4, iter=15, seed=seed)
        return model.wv

    @staticmethod
    def restrict_vectors(wordvectors, restricted_word_set):
        new_vectors = []
        new_vocab = {}
        new_index2entity = []
        new_vectors_norm = []
        wordvectors.init_sims()
        for i in range(len(wordvectors.vocab)):
            word = wordvectors.index2entity[i]
            vec = wordvectors.vectors[i]
            vocab = wordvectors.vocab[word]
            vec_norm = wordvectors.vectors_norm[i]
            if word in restricted_word_set:
                vocab.index = len(new_index2entity)
                new_index2entity.append(word)
                new_vocab[word] = vocab
                new_vectors.append(vec)
                new_vectors_norm.append(vec_norm)

        wordvectors.vocab = new_vocab
        wordvectors.vectors = np.array(new_vectors)
        wordvectors.index2entity = new_index2entity
        wordvectors.index2word = new_index2entity
        wordvectors.vectors_norm = new_vectors_norm

    @staticmethod
    def save(word_vectors: gensim.KeyedVectors, path="E:/AML4DHGermanVecs/test_vecs.kv"):
        word_vectors.save(get_tmpfile(path))

    @staticmethod
    def load(path="E:/AML4DHGermanVecs/test_vecs.kv") -> gensim.KeyedVectors:
        return gensim.KeyedVectors.load(path)
