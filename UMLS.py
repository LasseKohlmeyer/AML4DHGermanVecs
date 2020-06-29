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
    def __init__(self, from_dir: str = None, json_path: str = "mapper.json", umls_words: Iterable[str] = None):
        # self.db = DictDatabase(WordNgramFeatureExtractor(2))

        self.db = DictDatabase(CharacterNgramFeatureExtractor(2))

        if from_dir:
            json_path = os.path.join(from_dir, json_path)
            if os.path.exists(json_path):
                print(f"initialize {self.__class__.__name__}... Load json")
                self.umls_dict, self.umls_reverse_dict = self.load_from_json(json_path)
                self.add_words_to_db(self.umls_dict.keys())
            else:
                print(f"initialize {self.__class__.__name__}... Load dir")
                self.umls_dict, self.umls_reverse_dict = self.load_umls_dict(from_dir)
                self.add_words_to_db(self.umls_dict.keys())
                self.save_as_json(path=json_path)
        else:
            self.add_words_to_db(umls_words)

        # if from_dir:
        #     print(f"initialize {self.__class__.__name__}... Load dir")
        #     self.umls_dict, self.umls_reverse_dict = self.load_umls_dict(from_dir)
        #     self.add_words_to_db(self.umls_dict.keys())
        # elif json_path:
        #     print(f"initialize {self.__class__.__name__}... Load json")
        #     self.umls_dict, self.umls_reverse_dict = self.load_from_json(json_path)
        #     self.add_words_to_db(self.umls_dict.keys())
        # else:
        #     self.add_words_to_db(umls_words)

    def load_umls_dict(self, directory):
        path = os.path.join(directory, "GER_MRCONSO.RRF")
        df = pd.read_csv(path, delimiter="|", header=None)
        df.columns = ["CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI", "SAUI", "SCUI", "SDUI", "SAB", "TTY",
                      "CODE", "STR", "SRL", "SUPPRESS", "CVF", "NONE"]
        df = df.drop(columns=['NONE'])
        dic = {row["STR"]: row["CUI"] for i, row in df.iterrows()}
        rev_dic = defaultdict(list)
        for key, value in dic.items():
            rev_dic[value].append(key)
        return dic, rev_dic

    def save_as_json(self, path: str):
        data = self.__dict__
        data.pop("db")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=0)

    def load_from_json(self, path: str):
        with open(path, 'r', encoding='utf-8') as file:
            data = json.loads(file.read())
        return data["umls_dict"], data["umls_reverse_dict"]

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

    def replace_documents_token_based(self, documents: List[str], delete_non_umls=False) -> List[List[str]]:
        tokenized_documents = [sentence.split() for sentence in documents]
        return [[self.umls_code(token, delete_non_umls) for token in tokens if self.umls_code(token, delete_non_umls)]
                for tokens in tokenized_documents]

    def spacy_tokenize(self, documents: List["str"], nlp=None) -> List[List[str]]:
        if nlp is None:
            nlp = spacy.load('de_core_news_sm')
        doc_pipe = list(nlp.pipe(documents, disable=["tagger", "parser", "ner"]))
        tokenized_docs = [[token.text for token in doc] for doc in tqdm(doc_pipe, desc="Tokenize", total=len(documents))]
        return tokenized_docs

    def replace_documents_with_spacy_multiterm(self, documents: List[str]) -> List[List[str]]:
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
        replaced_docs = self.spacy_tokenize(replaced_docs, nlp)
        # doc_pipe = list(nlp.pipe(replaced_docs, disable=["tagger", "parser", "ner"]))
        # replaced_docs = [[token.text for token in doc] for doc in tqdm(doc_pipe, desc="Tokenize", total=len(documents))]

        return replaced_docs


class UMLSEvaluator(EvaluationResource):
    def set_attributes(self, *args):
        self.concept2category, self.category2concepts = args

    def __init__(self, from_dir: str = None, json_path: str = "umls_eval.json"):
        self.concept2category, self.category2concepts = None, None
        self.check_for_json_and_parse(from_dir=from_dir, json_path=json_path)

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


class MRRELEvaluator(EvaluationResource):
    def set_attributes(self, *args):
        self.mrrel_cause, self.mrrel_association = args

    def __init__(self, from_dir: str = None, json_path: str = "umls_rel_eval.json"):
        self.mrrel_cause = None
        self.mrrel_association = None
        self.check_for_json_and_parse(from_dir=from_dir, json_path=json_path)

    def load_semantics(self, directory):
        path = os.path.join(directory, "MRREL.RRF")
        df = pd.read_csv(path, delimiter="|", header=None)
        df.columns = ["CUI1", "AUI1", "STYPE1", "REL", "CUI2", "AUI2", "STYPE2", "RELA", "RUI", "SRUI", "SAB", "SL",
                      "RG", "DIR", "SUPPRESS", "CVF", "NONE"]
        df = df.drop(columns=["AUI1", "REL", "STYPE1", "AUI2", "STYPE2", "RUI", "SRUI", "SAB", "SL",
                              "RG", "DIR", "SUPPRESS", "CVF", "NONE"])

        df_cause = df.loc[df['RELA'].isin(['induces', 'cause_of', 'causative_agent_of'])]
        print(df_cause.head(100))

        mrrel_cause = defaultdict(list)

        for i, row in tqdm(df_cause.iterrows(), total=len(df_cause), desc="Find causative data"):
            mrrel_cause[row["CUI1"]].append(row["CUI2"])

        df_association = df.loc[df['RELA'].isin(['associated_disease', 'associated_finding_of',
                                                 'clinically_associated_with'])]
        print(df_association.head(100))

        mrrel_association = defaultdict(list)

        for i, row in tqdm(df_cause.iterrows(), total=len(df_association), desc="Find association data"):
            mrrel_association[row["CUI1"]].append(row["CUI2"])

        return mrrel_cause, mrrel_association

    def load_from_json(self, path: str):
        with open(path, 'r', encoding='utf-8') as file:
            data = json.loads(file.read())
        return data["mrrel_cause"], data["mrrel_association"]