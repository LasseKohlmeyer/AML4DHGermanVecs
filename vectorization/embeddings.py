import os
from collections import defaultdict
from multiprocessing.spawn import freeze_support
from typing import List, Dict, Union
import gensim
import torch
from flair.data import Dictionary, Sentence
from flair.embeddings import TransformerWordEmbeddings, FlairEmbeddings
from flair.trainers.language_model_trainer import TextCorpus, LanguageModelTrainer
from gensim.test.utils import get_tmpfile
from gensim.models import Phrases
import glove
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from numpy import float32 as real
from gensim import utils
from ..resource.UMLS import UMLSMapper
from ..utils.transform_data import DataHandler
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


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
        print("Started vectorization")
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
        try:
            word_vectors.save(get_tmpfile(path))
        except FileNotFoundError:
            word_vectors.save(path)

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

    @staticmethod
    def assign_concepts_to_vecs(vectors: gensim.models.KeyedVectors, umls_mapper: UMLSMapper):
        addable_concepts = []
        addable_vectors = []
        for concept, terms in umls_mapper.umls_reverse_dict.items():
            concept_vec = []
            for term in terms:
                term_tokens = term.split()
                token_vecs = []
                for token in term_tokens:
                    if token in vectors.vocab:
                        token_vecs.append(vectors.get_vector(token))
                if len(term_tokens) == len(token_vecs):
                    term_vector = sum(token_vecs)
                    concept_vec.append(term_vector)
            if len(concept_vec) > 0:
                addable_concepts.append(concept)
                addable_vectors.append(sum(concept_vec) / len(concept_vec))
        vectors.add(addable_concepts, addable_vectors)
        print(len(addable_concepts))
        return vectors

    @staticmethod
    def sentence_data2vec(path: Union[str, List[str]], embedding_name: str,
                          embeddings_algorithm: Union[str, gensim.models.Word2Vec, gensim.models.FastText] = "word2vec",
                          number_sentences: int = 1000,
                          use_phrases: bool = False,
                          restrict_vectors: bool = False,
                          umls_replacement: bool = True,
                          umls_path: str = 'E:/AML4DH-DATA/UMLS',
                          use_multiterm_replacement: bool = True,
                          flair_model_path: str = None,
                          flair_corpus_path: str = None,
                          flair_algorithm: str = 'de-forward'
                          ):
        is_flair = False
        if isinstance(embeddings_algorithm, str) and embeddings_algorithm.lower() == "word2vec":
            embeddings_algorithm = gensim.models.Word2Vec
        if isinstance(embeddings_algorithm, str) and embeddings_algorithm.lower() == "fasttext":
            embeddings_algorithm = gensim.models.FastText
        if isinstance(embeddings_algorithm, str) and embeddings_algorithm.lower() == "glove":
            embeddings_algorithm = Embeddings.glove_vectors
        if isinstance(embeddings_algorithm, str) and embeddings_algorithm.lower() == "flair":
            is_flair = True

        umls_mapper = UMLSMapper(from_dir=umls_path)
        if isinstance(path, list):
            data_sentences = DataHandler.concat_path_sentences(path)
            # old:
            # data_sentences = []
            # for p in path:
            #     data_sentences.extend(DataHandler.lines_from_file(path=p))
        else:
            data_sentences = DataHandler.lines_from_file(path=path)
        if number_sentences:
            data_sentences = data_sentences[:number_sentences]
        print((data_sentences[:10]))

        # cpg_words = lines_from_file(path="E:/AML4DH-DATA/CPG-AMIA2020/Plain Text/cpg-tokens.txt")

        # Preprocessing
        # cpg_words = umls_mapper.standardize_words(cpg_words)
        # cpg_words = umls_mapper.replace_with_umls(cpg_words)
        # cpg_words = DataHandler.preprocess(cpg_words)
        # print((cpg_words[:100]))

        # data_sentences = umls_mapper.standardize_documents(data_sentences)
        # data_sentences = umls_mapper.replace_documents_with_umls(data_sentences)
        # sents = [data_sentences[20124], data_sentences[20139]]
        tokenize = True
        if flair_model_path:
            tokenize = False
        if umls_replacement:
            if use_multiterm_replacement:
                data_sentences = umls_mapper.replace_documents_with_spacy_multiterm(data_sentences, tokenize=tokenize)
            else:
                data_sentences = umls_mapper.replace_documents_token_based(data_sentences, tokenize=tokenize)
            print(data_sentences[:10])
        else:
            data_sentences = umls_mapper.spacy_tokenize(data_sentences)
            print(data_sentences[:10])
        # for s in data_sentences:
        #     print(s)
        # data_sentences = umls_mapper.replace_documents_with_umls_smart(data_sentences)
        # data_sentences = DataHandler.preprocess(documents=data_sentences, lemmatize=True, remove_stopwords=True)

        # vecs = Embeddings.calculate_vectors([cpg_words], use_phrases=False)

        if is_flair:
            vecs = Flair.get_flair_vectors(data_sentences,
                                           flair_model_path=flair_model_path,
                                           flair_algorithm=flair_algorithm,
                                           retrain_corpus_path=flair_corpus_path)
        else:
            vecs = Embeddings.calculate_vectors(data_sentences,
                                                use_phrases=use_phrases,
                                                embedding_algorithm=embeddings_algorithm)

        print(f'Got {len(vecs.vocab)} vectors for {len(data_sentences)} sentences')

        Embeddings.save_medical(vecs, embedding_name, umls_mapper, restrict=restrict_vectors)


class Flair:
    @staticmethod
    def determine_algorithm_from_string(flair_algorithm_string: str):
        if 'bert' in flair_algorithm_string:
            return TransformerWordEmbeddings, 'bert'
        else:
            return FlairEmbeddings, 'flair'

    @staticmethod
    def build_flair_corpus(sentences: List[str], root_path: str):
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        val_ratio = 0.2
        test_ratio = 0.2

        first_split = val_ratio + test_ratio
        second_split = test_ratio / first_split

        print('split data')
        train_sentences, val_sentences = train_test_split(sentences, test_size=first_split, random_state=42)
        val_sentences, test_sentences = train_test_split(val_sentences, test_size=second_split, random_state=42)

        train_splits = chunks(train_sentences, 5000)

        print('save splits')
        try:
            os.mkdir(root_path)
            os.mkdir(os.path.join(root_path, 'train'))
        except OSError:
            print("Creation of the directory %s failed" % root_path)

        for j, split in enumerate(train_splits):
            DataHandler.save(os.path.join(root_path, 'train', f'train_split_{j}.txt'), '\n'.join(split))
        DataHandler.save(os.path.join(root_path, 'valid.txt'), '\n'.join(val_sentences))
        DataHandler.save(os.path.join(root_path, 'test.txt'), '\n'.join(test_sentences))

        return sentences

    @classmethod
    def retrain_flair(cls, corpus_path: str, model_path_dest: str, flair_algorithm: str = 'de-forward',
                      epochs: int = 10):
        use_embedding, algorithm = cls.determine_algorithm_from_string(flair_algorithm_string=flair_algorithm)
        # instantiate an existing LM, such as one from the FlairEmbeddings
        model = use_embedding(flair_algorithm)
        if algorithm == 'bert':
            language_model = model.model
        else:
            language_model = model.lm

        # are you fine-tuning a forward or backward LM?
        try:
            is_forward_lm = language_model.is_forward_lm
        except AttributeError:
            is_forward_lm = True

        # todo: no support for finetuning BERT with Flair Library for now
        # get the dictionary from the existing language model
        dictionary: Dictionary = language_model.dictionary

        # get your corpus, process forward and at the character level
        corpus = TextCorpus(corpus_path,
                            dictionary,
                            is_forward_lm,
                            character_level=True)

        # use the model trainer to fine-tune this model on your corpus
        trainer = LanguageModelTrainer(language_model, corpus)

        trainer.train(model_path_dest,
                      sequence_length=10,
                      mini_batch_size=10,
                      learning_rate=20,
                      max_epochs=epochs,
                      patience=10,
                      checkpoint=True)

    @classmethod
    def get_flair_vectors(cls, raw_sentences: Union[List[str], List[List[str]]],
                          flair_model_path: str,
                          flair_algorithm: str,
                          retrain_corpus_path: str = None,
                          epochs: int = 10):
        freeze_support()

        # retrain
        if retrain_corpus_path:
            if not os.path.isdir(retrain_corpus_path):
                raw_sentences = cls.build_flair_corpus(raw_sentences, retrain_corpus_path)
            cls.retrain_flair(corpus_path=retrain_corpus_path, model_path_dest=flair_model_path,
                              flair_algorithm=flair_algorithm, epochs=epochs)
        if os.path.exists(os.path.dirname(flair_model_path)):
            flair_model_path = os.path.join(flair_model_path, 'best-lm.pt')

        use_embedding, _ = cls.determine_algorithm_from_string(flair_algorithm_string=flair_algorithm)

        embedding = use_embedding(flair_model_path)

        if any(isinstance(el, list) for el in raw_sentences):
            use_tokenizer = False
            raw_sentences = [' '.join(raw_sentence) for raw_sentence in raw_sentences if len(raw_sentence) > 0]
        else:
            use_tokenizer = True

        flair_sents = [Sentence(raw_sentence, use_tokenizer=use_tokenizer)
                       for raw_sentence in tqdm(raw_sentences,
                                                desc="Convert to flair",
                                                total=len(raw_sentences))
                       if raw_sentence != '' and len(raw_sentence) > 0]

        flair_sents = [flair_sent for flair_sent in flair_sents if flair_sent and len(flair_sent) > 0]

        # keyed_vecs_o = defaultdict(list)
        # for flair_sentence in tqdm(flair_sents, desc='Embed sentences', total=len(flair_sents)):
        #     embedding.embed(flair_sentence)
        #     for token in flair_sentence:
        #         keyed_vecs_o[token.text].append(token.embedding.cpu())
        # keyed_vecs_o = {key: np.array(torch.mean(torch.stack(vecs), 0).cpu()) for key, vecs in keyed_vecs_o.items()}

        keyed_vecs = {}
        for flair_sentence in tqdm(flair_sents, desc='Embed sentences', total=len(flair_sents)):
            try:
                embedding.embed(flair_sentence)
            except IndexError:
                continue
            for token in flair_sentence:
                if token.text in keyed_vecs:
                    cur, inc = keyed_vecs[token.text]
                    new_token_embedding = token.embedding.cpu()
                    # print(len(np.array(new_token_embedding)))
                    if new_token_embedding.size() == cur.size():
                        keyed_vecs[token.text] = (cur + (new_token_embedding - cur) / (inc + 1), inc + 1)
                else:
                    keyed_vecs[token.text] = (token.embedding.cpu(), 1)
            flair_sentence.clear_embeddings()
        keyed_vecs = {key: np.array(vecs[0]) for key, vecs in keyed_vecs.items()}
        keyed_vecs = {key: vecs for key, vecs in keyed_vecs.items() if len(vecs) != 0}
        for key, vec in keyed_vecs.items():
            if len(vec) != 3072:
                print(key, len(vec))
        return Embeddings.to_gensim_binary(keyed_vecs)

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