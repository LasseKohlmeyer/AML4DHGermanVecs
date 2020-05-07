from collections import defaultdict
from typing import List, Dict, Iterable
import spacy
from gensim import models as gensim
from gensim.models import Word2Vec, FastText, Phrases
from gensim.test.utils import get_tmpfile
from gensim.test.utils import datapath
import pandas as pd
from simstring.feature_extractor.character_ngram import CharacterNgramFeatureExtractor
from simstring.feature_extractor.word_ngram import WordNgramFeatureExtractor
from simstring.measure.cosine import CosineMeasure
from simstring.database.dict import DictDatabase
from simstring.searcher import Searcher


class UMLSMapper:
    def __init__(self, umls_words: Iterable[str] = None):
        # self.db = DictDatabase(WordNgramFeatureExtractor(2))
        self.db = DictDatabase(CharacterNgramFeatureExtractor(2))
        if umls_words:
            self.add_words_to_db(umls_words)

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


def to_vecs(tokens=List[str], use_phrases: bool = False, w2v_model=gensim.Word2Vec) -> gensim.KeyedVectors:
    if use_phrases:
        bigram_transformer = Phrases(tokens)
        model = w2v_model(bigram_transformer[tokens], size=100, window=10, min_count=1, workers=4)
    else:
        model = Word2Vec(tokens, size=300, window=5, min_count=1, workers=4, iter=15)
    return model.wv


def save_vecs(word_vectors: gensim.KeyedVectors, path="E:/AML4DHGermanVecs/test_vecs.kv"):
    word_vectors.save(get_tmpfile(path))


def load_vecs(path="E:/AML4DH-DATA/AML4DHGermanVecs/test_vecs.kv") -> gensim.KeyedVectors:
    return gensim.KeyedVectors.load(path)


def get_UMLS_dict(path='E:/AML4DH-DATA/UMLS/GER_MRCONSO.RRF'):
    df = pd.read_csv(path, delimiter="|", header=None)
    df.columns = ["CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI", "SAUI", "SCUI", "SDUI", "SAB", "TTY",
                  "CODE", "STR", "SRL", "SUPPRESS", "CVF", "NONE"]
    dic = {row["STR"]: row["CUI"] for id, row in df.iterrows()}
    rev_dic = defaultdict(list)
    for key, value in dic.items():
        rev_dic[value].append(key)
    return dic, rev_dic


def umls_code(umls_dict, token, delete_non_umls=False):
    umls_code = umls_dict.get(token)
    if umls_code == None:
        if delete_non_umls:
            return None
        else:
            return token
    else:
        return umls_code


def un_umls(reverse_umls_dict, concept):
    res = reverse_umls_dict.get(concept)
    if res is None:
        return concept
    return res

def standardize_words(concept_dict, words):
    standardized_tokens = []
    for token in words:
        mapping = concept_dict.get(token)
        if mapping:
            standardized_tokens.append(mapping[0])
        else:
            standardized_tokens.append(token)
    return standardized_tokens

def replace_with_UMLS(umls_dict, tokens):
    return [umls_code(umls_dict, token) for token in tokens if umls_code(umls_dict, token)]


pre_tokens = tokens_from_file(path="E:/AML4DH-DATA/CPG-AMIA2020/Plain Text/cpg-tokens.txt")
umls, rev_umls = get_UMLS_dict()
umls_mapper = UMLSMapper(umls_words=umls.keys())
mapped_concepts = umls_mapper.search_all_term_sims(pre_tokens)
print(len(mapped_concepts))


tokens = standardize_words(mapped_concepts, pre_tokens)
tokens = preprocess(
    replace_with_UMLS(umls, tokens))
print((tokens[:100]))
vecs = to_vecs([tokens], use_phrases=True)
for c, v in vecs.most_similar(umls["Zervix"]):
    print(un_umls(rev_umls, c), v)
save_vecs(vecs)

# vecs = load_vecs()
# print(vecs.most_similar("Tumor"))
