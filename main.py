from collections import defaultdict
from typing import List, Dict, Iterable
import spacy
import os
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
from tqdm import tqdm
from whatlies.embedding import Embedding
from whatlies.embeddingset import EmbeddingSet
from itertools import chain


class UMLSMapper:
    # https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.T.concept_names_and_sources_file_mr/
    def __init__(self, from_dir=None, umls_words: Iterable[str] = None):
        # self.db = DictDatabase(WordNgramFeatureExtractor(2))
        self.db = DictDatabase(CharacterNgramFeatureExtractor(2))

        if from_dir:
            self.umls_dict, self.umls_reverse_dict = self.load_umls_dict(directory=from_dir)
            self.add_words_to_db(self.umls_dict.keys())
        else:
            self.add_words_to_db(umls_words)

    def load_umls_dict(self, directory: str):
        path = os.path.join(directory, "GER_MRCONSO.RRF")
        df = pd.read_csv(path, delimiter="|", header=None)
        df.columns = ["CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI", "SAUI", "SCUI", "SDUI", "SAB", "TTY",
                      "CODE", "STR", "SRL", "SUPPRESS", "CVF", "NONE"]
        df = df.drop(columns=['NONE'])
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

    def standardize_documents(self, documents: List[List[str]]):
        concept_dict = self.search_all_term_sims(list(chain(*documents)))
        standardized_documents = []
        for document in tqdm(documents):
            standardized_tokens = []
            for token in document:
                mapping = concept_dict.get(token)
                if mapping:
                    standardized_tokens.append(mapping[0])
                else:
                    standardized_tokens.append(token)
                standardized_documents.append(standardized_tokens)
        return standardized_documents

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

    def replace_umls(self, tokens: List[str]) -> List[str]:
        return [self.un_umls(token) for token in tokens if self.un_umls(token)]

    def umls_code(self, token, delete_non_umls):
        umls_code = self.umls_dict.get(token)
        if umls_code is None:
            if delete_non_umls:
                return None
            else:
                return token
        else:
            return umls_code

    def replace_with_UMLS(self, tokens: List[str], delete_non_umls=False) -> List[str]:
        return [self.umls_code(token, delete_non_umls) for token in tokens if self.umls_code(token, delete_non_umls)]

    def replace_documents_with_UMLS(self, documents: List[List[str]], delete_non_umls=False) -> List[List[str]]:
        return [[self.umls_code(token, delete_non_umls) for token in tokens if self.umls_code(token, delete_non_umls)]
                for tokens in documents]


class UMLSEvaluator:
    def __init__(self, vectors: gensim.KeyedVectors, from_dir="E:/AML4DH-DATA/UMLS"):
        self.vocab = vectors.vocab
        self.concept2category, self.category2concepts = self.load_umls_semantics(directory=from_dir)

    def load_umls_semantics(self, directory):
        path = os.path.join(directory, "MRSTY.RRF")
        df = pd.read_csv(path, delimiter="|", header=None)
        df.columns = ["CUI", "TUI", "STN", "STY", "ATUI", "CVF", "NONE"]
        df = df.drop(columns=['NONE'])

        concept2category = defaultdict(list)
        category2concepts = defaultdict(list)

        for i, row in df.iterrows():
            if row["CUI"] in self.vocab:
                concept2category[row["CUI"]].append(row["STY"])
                category2concepts[row["STY"]].append(row["CUI"])
        return concept2category, category2concepts


def lines_from_file(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        data = f.read()

    return data.split("\n")


def preprocess(tokens: List[str] = None, documents: List[List[str]] = None, lemmatize: bool = False, lower: bool = False,
               pos_filter: list = None, remove_stopwords: bool = False,
               remove_punctuation: bool = False, lan_model=None) -> List[List[str]]:
    def token_representation(token):
        representation = str(token.lemma_) if lemmatize else str(token)
        if lower:
            representation = representation.lower()
        return representation

    nlp = spacy.load("de_core_news_sm") if lan_model is None else lan_model
    nlp.Defaults.stop_words |= {"der", "die", "das", "Der", "Die", "Das", "bei", "Bei", "In", "in"}

    if tokens:
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

    if documents:
        documents = [' '.join(doc) for doc in documents]
        for word in nlp.Defaults.stop_words:
            lex = nlp.vocab[word]
            lex.is_stop = True
        preprocessed_documents = []

        if pos_filter is None:
            for doc in nlp.pipe(documents, disable=['parser', 'ner', 'tagger']):
                preprocessed_document = []
                for token in doc:
                    if (not remove_stopwords or not token.is_stop) and (not remove_punctuation or token.is_alpha):
                        preprocessed_document.append(token_representation(token))
                preprocessed_documents.append(preprocessed_document)

        else:
            for doc in nlp.pipe(documents, disable=['parser', 'ner']):
                preprocessed_document = []
                for token in doc:
                    if (not remove_stopwords or not token.is_stop) and (
                            not remove_punctuation or token.is_alpha) and token.pos_ in pos_filter:
                        preprocessed_document.append(token_representation(token))
                preprocessed_documents.append(preprocessed_document)

        return preprocessed_documents


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


umls_mapper = UMLSMapper(from_dir='E:/AML4DH-DATA/UMLS')



# load sentences
# cpq_sentences = lines_from_file(path="E:/AML4DH-DATA/CPG-AMIA2020/Plain Text/cpg-sentences.txt")
#
# # tokenization
# cpq_sentences = [sentence.split() for sentence in cpq_sentences]
#
# cpg_words = lines_from_file(path="E:/AML4DH-DATA/CPG-AMIA2020/Plain Text/cpg-tokens.txt")
#
# # Preprocessing
# # cpg_words = umls_mapper.standardize_words(cpg_words)
# # cpg_words = umls_mapper.replace_with_UMLS(cpg_words)
# # cpg_words = preprocess(cpg_words)
# # print((cpg_words[:100]))
#
# # cpq_sentences = umls_mapper.standardize_documents(cpq_sentences)
# cpq_sentences = umls_mapper.replace_documents_with_UMLS(cpq_sentences)
# cpq_sentences = preprocess(documents=cpq_sentences, lemmatize=True, remove_stopwords=True)
#
# print((cpq_sentences[:10]))
#
#
# #
# # # Vectorization
# # vecs = Embeddings.calculate_vectors([cpg_words], use_phrases=False)
# vecs = Embeddings.calculate_vectors(cpq_sentences, use_phrases=False)
#
#
# # Embeddings.save(vecs)
# #
# # vecs = Embeddings.load()
# # print(vecs.vocab)
#
# # Evaluation
# for c, v in vecs.most_similar("Cisplatin"):
#     print(umls_mapper.un_umls(c), v)
# concept_vecs = umls_mapper.get_umls_vectors_only(vecs)
#
#
# # for c, v in vecs.most_similar(umls_mapper.umls_dict["Cisplatin"]):
# #     print(umls_mapper.un_umls(c), v)
# # concept_vecs = umls_mapper.get_umls_vectors_only(vecs)
#
#
# Embeddings.restrict_vectors(vecs, concept_vecs.keys())
# Embeddings.save(vecs, path="E:/AML4DHGermanVecs/test_vecs_1.kv")

vecs = Embeddings.load(path="/data/test_vecs_1.kv")

def analogies(vectors, start, minus, plus, umls: UMLSMapper):
    if umls:
        return vectors.most_similar(positive=[umls.umls_dict[start], umls.umls_dict[plus]], negative=[umls.umls_dict[minus]])
    else:
        return vectors.most_similar(positive=[start, plus], negative=[minus])

def similarities(vectors, word, umls):
    if umls:
        return vectors.most_similar(umls.umls_dict[word])
    else:
        return vectors.most_similar(word)

def similarities2(self, word):
    if self.umls:
        return self.most_similar(self.umls.umls_dict[word])
    else:
        return self.most_similar(word)

vecs.umls = umls_mapper
gensim.KeyedVectors.monkey_sims = similarities2
print(vecs.monkey_sims("Asthma"))
for c, v in analogies(vecs, "Asthma", "Lunge", "Herz", umls=umls_mapper):
    print(umls_mapper.un_umls(c, single_return=True), v)


for c, v in similarities(vecs, "Hepatitis", umls=umls_mapper):
    print(umls_mapper.un_umls(c, single_return=True), v)

for c, v in similarities(vecs, "Cisplatin", umls=umls_mapper):
    print(umls_mapper.un_umls(c, single_return=True), v)

# print([(umls_mapper.un_umls(c), Embedding(umls_mapper.un_umls(c), vecs[c])) for c in vecs.vocab])
def pairwise_cosine(concepts1, concepts2=None):
    def cosine(word1=None, word2=None, concept1=None, concept2=None):
        if word1:
            return vecs.similarity(umls_mapper.umls_dict[word1], umls_mapper.umls_dict[word2])
        else:
            return vecs.similarity(concept1, concept2)
    if concepts2 is None:
        concepts2 = concepts1
        s = 0
        count = 0
        for i, concept1 in enumerate(concepts1):
            for j, concept2 in enumerate(concepts2):
                if j > i:
                    c = cosine(concept1=concept1, concept2=concept2)
                    if c < 0:
                        c = -c
                    s += c
                    count += 1
        return s / count
    else:
        s = 0
        count = 0
        for i, concept1 in enumerate(concepts1):
            for j, concept2 in enumerate(concepts2):
                c = cosine(concept1=concept1, concept2=concept2)
                if c < 0:
                    c = -c
                s += c
                count += 1
        return s / count


def bla():
    choosen_category = "Nucleotide Sequence"
    other_categories = evaluator.category2concepts.keys()
    p1 = pairwise_cosine(evaluator.category2concepts[choosen_category])
    print(p1)

    p2s = []
    for other_category in other_categories:
        if other_category == choosen_category:
            continue
        choosen_concepts = evaluator.category2concepts[choosen_category]
        other_concepts = evaluator.category2concepts[other_category]
        if len(choosen_concepts) == 0 or len(other_concepts) == 0:
            continue
        p2 = pairwise_cosine(choosen_concepts, other_concepts)
        p2s.append(p2)

    avg_p2 = sum(p2s) / len(p2s)
    print(p2s)
    print(p1, avg_p2, p1 - avg_p2)
    return p1 - avg_p2


bla()

evaluator = UMLSEvaluator(vectors=vecs)
p1 = pairwise_cosine(evaluator.category2concepts["Medical Device"])
print(p1)
p2 = pairwise_cosine(evaluator.category2concepts["Medical Device"], evaluator.category2concepts["Health Care Related Organization"])
print(p2)

emb = EmbeddingSet({umls_mapper.un_umls(c, single_return=True): Embedding(umls_mapper.un_umls(c, single_return=True), vecs[c]) for c in vecs.vocab})
# emb = EmbeddingSet({c: Embedding(c, vecs[c]) for c in vecs.vocab})

emb.plot_interactive("Fibroblasten","Fremdkörper")

# replace multi words