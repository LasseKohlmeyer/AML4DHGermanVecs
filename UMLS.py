import os
from collections import defaultdict
from typing import Iterable, List, Dict
import json
import gensim
from simstring.database.dict import DictDatabase
from simstring.feature_extractor.character_ngram import CharacterNgramFeatureExtractor
from simstring.measure.cosine import CosineMeasure
from simstring.searcher import Searcher
from tqdm import tqdm
import pandas as pd
from itertools import chain
import spacy
from spacy.matcher import PhraseMatcher

from evaluation_resource import EvaluationResource


class UMLSMapper:
    # https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.T.concept_names_and_sources_file_mr/
    def __init__(self, from_dir=None, umls_words: Iterable[str] = None):
        # self.db = DictDatabase(WordNgramFeatureExtractor(2))
        print(f"initialize {self.__class__.__name__}...")
        self.db = DictDatabase(CharacterNgramFeatureExtractor(2))

        if from_dir:
            self.directory = from_dir
            self.umls_dict, self.umls_reverse_dict = self.load_umls_dict()
            self.add_words_to_db(self.umls_dict.keys())
        else:
            self.add_words_to_db(umls_words)

    def load_umls_dict(self):
        path = os.path.join(self.directory, "GER_MRCONSO.RRF")
        df = pd.read_csv(path, delimiter="|", header=None)
        df.columns = ["CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI", "SAUI", "SCUI", "SDUI", "SAB", "TTY",
                      "CODE", "STR", "SRL", "SUPPRESS", "CVF", "NONE"]
        df = df.drop(columns=['NONE'])
        dic = {row["STR"]: row["CUI"] for i, row in df.iterrows()}
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

    def get_umls_vectors_only(self, vectors: gensim.models.KeyedVectors):
        medical_concepts = [word for word in vectors.index2word if word in self.umls_dict.values()]
        concept_vecs = {concept: vectors.get_vector(concept) for concept in medical_concepts}
        return concept_vecs

    def un_umls(self, concept, single_return=True):
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

    def replace_with_umls(self, tokens: List[str], delete_non_umls=False) -> List[str]:
        return [self.umls_code(token, delete_non_umls) for token in tokens if self.umls_code(token, delete_non_umls)]

    def replace_documents_with_umls(self, documents: List[str], delete_non_umls=False) -> List[List[str]]:
        tokenized_documents = [sentence.split() for sentence in documents]
        return [[self.umls_code(token, delete_non_umls) for token in tokens if self.umls_code(token, delete_non_umls)]
                for tokens in tokenized_documents]

    def replace_documents_with_spacy(self, documents: List[str]) -> List[List[str]]:
        nlp = spacy.load('de_core_news_sm')
        matcher = PhraseMatcher(nlp.vocab)
        terms = self.umls_dict.keys()
        doc_pipe = list(nlp.pipe(documents, disable=["tagger", "parser", "ner"]))
        # Only run nlp.make_doc to speed things up
        patterns = [nlp.make_doc(term) for term in terms]
        matcher.add("TerminologyList", None, *patterns)
        replaced_docs = []

        for doc in tqdm(doc_pipe, desc="Replace with concepts", total=len(documents)):
            text_doc = doc.text
            matches = matcher(doc)
            concepts = []
            for match_id, start, end in matches:
                span = doc[start:end]
                concepts.append(span.text)

            concepts.sort(key=lambda s: len(s), reverse=True)
            for concept in concepts:
                text_doc = text_doc.replace(concept, self.umls_dict[concept])


            replaced_docs.append(text_doc)

            # tokens = [token for token in text_doc.split()]
            # replaced_docs.append(tokens)

        doc_pipe = list(nlp.pipe(replaced_docs, disable=["tagger", "parser", "ner"]))
        replaced_docs = [[token.text for token in doc] for doc in tqdm(doc_pipe, desc="Tokenize", total=len(documents)) ]

        return replaced_docs


class UMLSEvaluator(EvaluationResource):
    def __init__(self, json_path: str = None, from_dir: str = None):

        if from_dir:
            print(f"initialize {self.__class__.__name__}... Load dir")
            self.concept2category, self.category2concepts = self.load_semantics(directory=from_dir)
        if json_path:
            print(f"initialize {self.__class__.__name__}... Load json")
            self.concept2category, self.category2concepts = self.load_from_json(json_path)

    def load_semantics(self, directory):
        path = os.path.join(directory, "MRSTY.RRF")
        df = pd.read_csv(path, delimiter="|", header=None)
        df.columns = ["CUI", "TUI", "STN", "STY", "ATUI", "CVF", "NONE"]
        df = df.drop(columns=['NONE'])
        concept2category = defaultdict(list)
        category2concepts = defaultdict(list)

        for i, row in tqdm(df.iterrows(), total=len(df), desc="Load UMLS data"):
            concept2category[row["CUI"]].append(row["STY"])
            category2concepts[row["STY"]].append(row["CUI"])
        return concept2category, category2concepts

    def load_from_json(self, path: str):
        with open(path, 'r', encoding='utf-8') as file:
            data = json.loads(file.read())
        return data["concept2category"], data["category2concepts"]
