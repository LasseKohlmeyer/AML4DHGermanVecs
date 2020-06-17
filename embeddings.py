from collections import defaultdict, OrderedDict
from typing import List, Dict
import logging
import gensim
from gensim.test.utils import get_tmpfile
from gensim.models import Phrases, Word2Vec, FastText
import glove
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
from tqdm import tqdm

from numpy import float32 as real
from gensim import utils


class Embeddings:
    @staticmethod
    def to_gensim_binary(dict_vecs: Dict[str, np.ndarray]) -> gensim.models.KeyedVectors:
        def my_save_word2vec_format(fname: str, vocab: Dict[str, np.ndarray], vectors: np.ndarray, binary: bool = True,
                                    total_vec: int = 2):
            """Store the input-hidden weight matrix in the same format used by the original
            C word2vec-tool, for compatibility.

            Parameters
            ----------
            fname : str
                The file path used to save the vectors in.
            vocab : dict
                The vocabulary of words.
            vectors : numpy.array
                The vectors to be stored.
            binary : bool, optional
                If True, the data wil be saved in binary word2vec format, else it will be saved in plain text.
            total_vec : int, optional
                Explicitly specify total number of vectors
                (in case word vectors are appended with document vectors afterwards).

            """
            if not (vocab or vectors):
                raise RuntimeError("no input")
            if total_vec is None:
                total_vec = len(vocab)
            vector_size = vectors.shape[1]
            assert (len(vocab), vector_size) == vectors.shape
            with utils.open(fname, 'wb') as fout:
                fout.write(utils.to_utf8("%s %s\n" % (total_vec, vector_size)))
                # store in sorted order: most frequent words at the top
                for word, row in vocab.items():
                    if binary:
                        row = row.astype(real)
                        fout.write(utils.to_utf8(word) + b" " + row.tostring())
                    else:
                        fout.write(utils.to_utf8("%s %s\n" % (word, ' '.join(repr(val) for val in row))))

        file_name = 'data/train.bin'
        dim = 0
        for vec in dict_vecs.values():
            dim = len(vec)
            break
        m = gensim.models.keyedvectors.Word2VecKeyedVectors(vector_size=dim)
        m.vocab = dict_vecs
        m.vectors = np.array(list(dict_vecs.values()))
        # print(m.vocab)
        my_save_word2vec_format(binary=True, fname=file_name, total_vec=len(dict_vecs), vocab=m.vocab,
                                vectors=m.vectors)
        reloaded_vecs = gensim.models.keyedvectors.Word2VecKeyedVectors.load_word2vec_format(file_name, binary=True,
                                                                                             unicode_errors="replace")
        return reloaded_vecs

    @staticmethod
    def glove_vectors(sentences: List[List[str]],
                      window_size: int = 5,
                      dim: int = 300,
                      epochs: int = 15,
                      step_size: float = 0.05,
                      workers: int = 12,
                      batch_size: int = 50,
                      seed: int = 42,
                      alpha: float = 0.025,
                      x_max: int = 100) -> gensim.models.KeyedVectors:

        def nan_checker(inp):
            if np.isnan(inp) or inp == np.nan:
                return 0.00000001
            else:
                return inp

        def build_co_occurrence_dict():
            d = defaultdict(lambda: defaultdict(int))
            vocab = {}
            reverse_vocab = {}
            tokenized_sentences = []
            id_counter = 0
            filtered_sentences = [sent for sent in sentences if len(sent) > 0]
            for text in filtered_sentences:
                # preprocessing (use tokenizer instead)
                if isinstance(text, str):
                    text = text.split()

                # while len(text) < sentence_length:
                #     text.insert(0, "%SoS%")
                # text = text[:sentence_length]

                tokenized_sentences.append(text)

                for token in text:
                    if token not in vocab:
                        vocab[token] = id_counter
                        reverse_vocab[id_counter] = token
                        id_counter += 1

            # print('>', vocab)
            # print('>>', reverse_vocab)
            # print(tokenized_sentences)

            for text_tokens in tokenized_sentences:
                for i in range(len(text_tokens)):
                    token = text_tokens[i]
                    next_token = text_tokens[i + 1: i + 1 + window_size]
                    for t in next_token:
                        key = tuple(sorted([t, token]))
                        d[vocab[key[0]]][vocab[key[1]]] += 1
                        d[vocab[key[1]]][vocab[key[0]]] += 1

            # for key, sub_dict in d.items():
            #     for sub_key, value in sub_dict.items():
            #         d[sub_key][key] = value
            for key in reverse_vocab:
                if key not in d:
                    d[key][key] = 1

            cooccur = {k: {k_i: v_i for k_i, v_i in v.items()} for k, v in d.items()}
            # cooccur = {k: {k_i: v_i for k_i, v_i in OrderedDict(sorted(v.items())).items()}
            #            for k, v in OrderedDict(sorted(d.items())).items()}
            # print(cooccur)
            return cooccur, vocab, reverse_vocab

        cooccurrence_dict, vocabulary, reverse_vocabulary = build_co_occurrence_dict()

        # print(cooccurrence_dict)
        # print(dim)
        model = glove.Glove(cooccurrence_dict, d=dim, alpha=alpha, x_max=x_max, seed=seed)

        epoch_bar = tqdm(range(epochs))
        for epoch in epoch_bar:
            err = model.train(step_size=step_size, workers=workers, batch_size=batch_size)
            epoch_bar.set_description("Glove epoch %d, error %.10f" % (epoch+1, err))
            epoch_bar.update()

        vecs = [np.array([nan_checker(ele) for ele in vec]) for vec in model.W]

        # print(len(model.W[0]))
        # print(len(vocabulary), len(reverse_vocabulary))

        # for key in (set(vocabulary.keys()).difference(set(reverse_vocabulary.values()))):
        #     print(key, vocabulary[key])

        return Embeddings.to_gensim_binary({word: vector for word, vector in zip(vocabulary.keys(), vecs)})

    @staticmethod
    def calculate_vectors(sentences=List[str],
                          use_phrases: bool = False,
                          embedding_algorithm=gensim.models.Word2Vec,
                          dim: int = 300,
                          window: int = 10,
                          seed: int = 42,
                          workers: int = 12,
                          epochs: int = 15,
                          min_count: int = 1,
                          alpha: float = 0.025,
                          step_size: float = 0.05,
                          batch_size: int = 50,
                          x_max: int = 100) -> gensim.models.KeyedVectors:
        logging.info("Started vectorization")
        if embedding_algorithm == Embeddings.glove_vectors:
            return Embeddings.glove_vectors(sentences=sentences,
                                            window_size=window,
                                            dim=dim,
                                            epochs=epochs,
                                            seed=seed,
                                            workers=workers,
                                            step_size=step_size,
                                            batch_size=batch_size,
                                            alpha=alpha,
                                            x_max=x_max)
        if use_phrases:
            bigram_transformer = Phrases(sentences)
            model = embedding_algorithm(bigram_transformer[sentences], size=dim, window=window, min_count=min_count,
                                        workers=workers, seed=seed, iter=epochs, alpha=alpha)
        else:
            model = embedding_algorithm(sentences, size=dim, window=window, min_count=min_count, workers=workers,
                                        iter=epochs, seed=seed, alpha=alpha)
        return model.wv

    @staticmethod
    def restrict_vectors(wordvectors: gensim.models.KeyedVectors, restricted_word_set):
        new_vectors = []
        new_vocab = {}
        new_index2entity = []
        new_vectors_norm = []
        wordvectors.init_sims()
        for i in tqdm(range(len(wordvectors.vocab)), desc="Vector restriction", total=len(wordvectors.vocab)):
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
    def save(word_vectors: gensim.models.KeyedVectors, path: str):
        word_vectors.save(get_tmpfile(path))

    @staticmethod
    def save_medical(word_vectors: gensim.models.KeyedVectors, name: str, umls_mapping, restrict=True):
        Embeddings.save(word_vectors, path=f"E:/AML4DHGermanVecs/data/{name}_all.kv")
        print(f'saved {name} vectors')
        if restrict:
            concept_vecs = umls_mapping.get_umls_vectors_only(word_vectors)
            Embeddings.restrict_vectors(word_vectors, concept_vecs.keys())
            Embeddings.save(word_vectors, path=f"E:/AML4DHGermanVecs/data/{name}.kv")
            print(f'Restricted to {len(word_vectors.vocab)} vectors')

    @staticmethod
    def load(path: str) -> gensim.models.KeyedVectors:
        print(f"load embedding of file {path}...")
        return gensim.models.KeyedVectors.load(path)

    @staticmethod
    def load_w2v_format(path: str, binary=False) -> gensim.models.KeyedVectors:
        print(f"load embedding of file {path}...")
        return gensim.models.KeyedVectors.load_word2vec_format(path, binary=binary)

    @staticmethod
    def transform_glove_in_word2vec(glove_input_file: str, word2vec_output_file: str):
        glove2word2vec(glove_input_file, word2vec_output_file)

# # sents = [" ".join(['C0850666', 'ist', 'der', 'wesentliche', 'Risikofaktor', 'für', 'das', 'C0699791', '.']),
# #          " ".join(['Die', 'H.', 'pylori-Eradikation', 'mit', 'dem', 'Ziel', 'der', 'Magenkarzinomprävention', 'sollte', 'bei',
# #           'den', 'folgenden', 'Risikopersonen', 'durchgeführt', 'werden', '(', 'siehe', 'Tabelle', 'unten', ')', '.'])]
# # sents = [" ".join(["ente", "futter", "gans", "bier"]), " ".join(["ente", "gans", "another", "esel", "der"])]
# # print()
# # print(sents)
# # sents = ["ente futter gans bier", "ente gans another esel der"]
# sents = [['Bei', '', 'Malignompatienten', 'einem', 'mehreren', 'dieser', 'C0035648', 'in', 'der', 'Regel', 'die', 'prophylaktische', 'Gabe'],
# ['Die', 'C0086818', 'wird', 'bei', 'hämatologisch-onkologischen', 'onkologischen', 'C0030705', 'mit', 'akuter', 'Thrombozytenbildungsstörung', 'und', 'zusätzlichen', 'Blutungsrisiken', 'empfohlen', 'bei', ':'], ['Thrombozytenkonzentrate', '(', 'TK', ')', 'werden', 'entweder', 'aus', 'Vollblutspenden', 'oder', 'durch', 'C0032202', 'von', 'gesunden', 'Blutspendern', 'gewonnen', '.'], ['Falls', 'nicht', 'verfügbar', ',', 'sollte', 'eine', 'entsprechende', 'C0010210', 'stattfinden', 'oder', 'Kontaktadressen', 'vermittelt', 'werden', '.'], ['M.', 'Meissner', ',', 'W.', 'Nehls', ',', 'J.', 'Gärtner', ',', 'U.', 'Kleeberg', ',', 'R.', 'Voltz'], [' '], ['C3816218', 'ist', 'definiert', 'als', 'ein', 'Ansatz', 'zur', 'Verbesserung', 'der', 'C0034380', 'von', 'C0030705', 'und', 'ihren', 'Familien', ',', 'die', 'mit', 'Problemen', 'konfrontiert', 'sind', ',', 'welche', 'mit', 'einer', 'lebensbedrohlichen', 'Erkrankung', 'einhergehen', '.']]
# print(sents)
# # sents = [" ".join(['C0850666', 'ist', 'der', 'wesentliche', 'Risikofaktor', 'für', 'das', 'C0699791', '.']),
# #          " ".join(['Die', 'H.', 'pylori-Eradikation', 'mit', 'dem', 'Ziel'])]
# # sents = [['C0850666', 'ist', 'der', 'wesentliche', 'Risikofaktor', 'für', 'das', 'C0699791', '.'],
# #          ['Die', 'H.', 'pylori-Eradikation', 'mit', 'dem', 'Ziel']]
#
# # print(len(['Die', 'H.', 'pylori-Eradikation', 'mit', 'dem', 'Ziel', 'der', 'Magenkarzinomprävention', 'sollte', 'bei',
# #           'den', 'folgenden', 'Risikopersonen', 'durchgeführt', 'werden', '(', 'siehe', 'Tabelle', 'unten', ')', '.']))
# kv = Embeddings.glove_vectors(sentences=sents)
# print(kv.vocab)
# # print(kv.most_similar("der"))