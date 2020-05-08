from collections import defaultdict
from typing import List, Dict, Iterable
import spacy
import numpy as np
from gensim import models as gensim
from gensim.models import Word2Vec, FastText, Phrases
from gensim.test.utils import get_tmpfile
import pandas as pd
from simstring.feature_extractor.character_ngram import CharacterNgramFeatureExtractor
from simstring.feature_extractor.word_ngram import WordNgramFeatureExtractor
from simstring.measure.cosine import CosineMeasure
from simstring.database.dict import DictDatabase
from simstring.searcher import Searcher
from whatlies.embedding import Embedding
from whatlies.embeddingset import EmbeddingSet


class UMLSMapper:
    # https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.T.concept_names_and_sources_file_mr/
    def __init__(self, from_file=None, umls_words: Iterable[str] = None):
        # self.db = DictDatabase(WordNgramFeatureExtractor(2))
        self.db = DictDatabase(CharacterNgramFeatureExtractor(2))

        if from_file:
            self.umls_dict, self.umls_reverse_dict = self.load_UMLS_dict(path=from_file)
            self.add_words_to_db(self.umls_dict.keys())
        else:
            self.add_words_to_db(umls_words)

    def load_UMLS_dict(self, path: str = 'E:/AML4DH-DATA/UMLS/GER_MRCONSO.RRF'):
        df = pd.read_csv(path, delimiter="|", header=None)
        df.columns = ["CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI", "SAUI", "SCUI", "SDUI", "SAB", "TTY",
                      "CODE", "STR", "SRL", "SUPPRESS", "CVF", "NONE"]
        dic = {row["STR"]: row["CUI"] for id, row in df.iterrows()}
        rev_dic = defaultdict(list)
        for key, value in dic.items():
            rev_dic[value].append(key)
        return dic, rev_dic

    def add_words_to_db(self, words: Iterable[str]):
        for token in set(words):
            self.db.add(token)

    def search_term_sims(self, term: str) -> List[str]:
        searcher = Searcher(self.db, CosineMeasure())
        return searcher.search(term, 0.8)

    def search_all_term_sims(self, terms: List[str]) -> Dict[str, List[str]]:
        dic = {}
        for term in set(terms):
            related_terms = self.search_term_sims(term)
            if len(related_terms) > 0:
                dic[term] = related_terms
        return dic
        # return {term: self.search_term_sims(term) for term in set(terms)}

    def standardize_words(self, tokens: List[str]):
        concept_dict = self.search_all_term_sims(tokens)
        standardized_tokens = []
        for token in tokens:
            mapping = concept_dict.get(token)
            if mapping:
                standardized_tokens.append(mapping[0])
            else:
                standardized_tokens.append(token)
        return standardized_tokens

    def get_umls_vectors_only(self, vectors: gensim.KeyedVectors):
        medical_concepts = [word for word in vectors.index2word if word in self.umls_dict.values()]
        print(len(medical_concepts), medical_concepts[:10])
        concept_vecs = {concept: vectors.get_vector(concept) for concept in medical_concepts}
        return concept_vecs

    def un_umls(self, concept, single_return=False):
        res = self.umls_reverse_dict.get(concept)
        if res is None:
            return concept

        if single_return:
            return res[0]
        else:
            return res

    def replace_UMLS(self, tokens: List[str]) -> List[str]:
        return [self.un_umls(token) for token in tokens if self.un_umls(token)]

    def replace_with_UMLS(self, tokens: List[str], delete_non_umls=False) -> List[str]:
        def umls_code(token):
            umls_code = self.umls_dict.get(token)
            if umls_code is None:
                if delete_non_umls:
                    return None
                else:
                    return token
            else:
                return umls_code

        return [umls_code(token) for token in tokens if umls_code(token)]


def tokens_from_file(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        data = f.read()

    return data.split("\n")


def preprocess(tokens: List[str], lemmatize: bool = True, lower: bool = False,
               pos_filter: list = None, remove_stopwords: bool = True,
               remove_punctuation: bool = False, lan_model=None) -> List[str]:
    def token_representation(token):
        representation = str(token.lemma_) if lemmatize else str(token)
        if lower:
            representation = representation.lower()
        return representation

    nlp = spacy.load("de_core_news_sm") if lan_model is None else lan_model
    nlp.Defaults.stop_words |= {"der", "die", "das", "Der", "Die", "Das", "bei", "Bei", "In", "in"}

    for word in nlp.Defaults.stop_words:
        lex = nlp.vocab[word]
        lex.is_stop = True
    preprocessed_tokens = []

    if pos_filter is None:
        for doc in nlp.pipe(tokens, disable=['parser', 'ner', 'tagger']):
            for token in doc:
                if (not remove_stopwords or not token.is_stop) and (not remove_punctuation or token.is_alpha):
                    preprocessed_tokens.append(token_representation(token))

    else:
        for doc in nlp.pipe(tokens, disable=['parser', 'ner']):
            for token in doc:
                if (not remove_stopwords or not token.is_stop) and (
                        not remove_punctuation or token.is_alpha) and token.pos_ in pos_filter:
                    preprocessed_tokens.append(token_representation(token))
    return preprocessed_tokens


class Embeddings:
    @staticmethod
    def calculate_vectors(tokens=List[str], use_phrases: bool = False,
                          w2v_model=gensim.Word2Vec) -> gensim.KeyedVectors:
        if use_phrases:
            bigram_transformer = Phrases(tokens)
            model = w2v_model(bigram_transformer[tokens], size=100, window=10, min_count=1, workers=4)
        else:
            model = Word2Vec(tokens, size=300, window=5, min_count=1, workers=4, iter=15)
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


umls_mapper = UMLSMapper(from_file='E:/AML4DH-DATA/UMLS/GER_MRCONSO.RRF')

# cpg_words = tokens_from_file(path="E:/AML4DH-DATA/CPG-AMIA2020/Plain Text/cpg-tokens.txt")
#
# # Preprocessing
# cpg_words = umls_mapper.standardize_words(cpg_words)
# cpg_words = umls_mapper.replace_with_UMLS(cpg_words)
# cpg_words = preprocess(cpg_words)
# print((cpg_words[:100]))
#
# # Vectorization
# vecs = Embeddings.calculate_vectors([cpg_words], use_phrases=True)
#
#
# Embeddings.save(vecs)
#
# vecs = Embeddings.load()
# # Evaluation
# for c, v in vecs.most_similar(umls_mapper.umls_dict["Zervix"]):
#     print(umls_mapper.un_umls(c), v)
# concept_vecs = umls_mapper.get_umls_vectors_only(vecs)
#
#
# Embeddings.restrict_vectors(vecs, concept_vecs.keys())
# Embeddings.save(vecs, path="E:/AML4DHGermanVecs/test_vecs_1.kv")

vecs = Embeddings.load(path="E:/AML4DHGermanVecs/test_vecs_1.kv")
for c, v in vecs.most_similar(umls_mapper.umls_dict["Zervix"]):
    print(umls_mapper.un_umls(c, single_return=True), v)

# print([(umls_mapper.un_umls(c), Embedding(umls_mapper.un_umls(c), vecs[c])) for c in vecs.vocab])
emb = EmbeddingSet({umls_mapper.un_umls(c, single_return=True): Embedding(umls_mapper.un_umls(c, single_return=True), vecs[c]) for c in vecs.vocab})
# emb = EmbeddingSet({c: Embedding(c, vecs[c]) for c in vecs.vocab})

emb.plot_interactive("Fibroblasten","Fremdkörper")
