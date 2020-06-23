import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import Tuple, Dict, Set, Iterable, List, Union

import gensim
import numpy as np
from gensim import matutils
# from gensim.models.wrappers import FastText
from gensim.models.fasttext import load_facebook_model
from scipy.stats import mannwhitneyu, ttest_ind
from scipy.stats.mstats import spearmanr
from tqdm import tqdm

import constant
from UMLS import UMLSMapper, UMLSEvaluator
from evaluation_resource import NDFEvaluator, SRSEvaluator
from joblib import Parallel, delayed
import multiprocessing


def revert_list_dict(dictionary: Dict[str, Set[str]], filter_collection: Iterable = None) -> Dict[str, Set[str]]:
    reverted_dictionary = defaultdict(set)
    for key, values in dictionary.items():
        for value in values:
            if not filter_collection or value in filter_collection:
                reverted_dictionary[value].add(key)
    return reverted_dictionary


class Benchmark(ABC):
    def __init__(self, embeddings: Tuple[Union[gensim.models.KeyedVectors], str, str, str],
                 umls_mapper: UMLSMapper,
                 umls_evaluator: UMLSEvaluator = None):
        self.vectors, self.dataset, self.algorithm, self.preprocessing = embeddings
        try:
            self.vocab = self.vectors.vocab
        except AttributeError:
            self.vocab = self.vectors.vocabulary
        self.umls_mapper = umls_mapper
        if umls_evaluator:
            self.umls_evaluator = umls_evaluator

        self.oov_embedding = None

    @abstractmethod
    def evaluate(self) -> float:
        pass

    def cosine(self, word1: str = None, word2: str = None,
               concept1: str = None, concept2: str = None,
               vector1: np.ndarray = None, vector2: np.ndarray = None) -> Union[float, None]:
        cos = None

        if word1 and word2:
            cos = self.vectors.similarity(self.umls_mapper.umls_dict[word1], self.umls_mapper.umls_dict[word2])

        if concept1 and concept2:
            if concept1 in self.vocab and concept2 in self.vocab:
                cos = self.vectors.similarity(concept1, concept2)
            else:

                vector1 = self.get_concept_vector(concept1)
                vector2 = self.get_concept_vector(concept2)
                # print(concept1, concept2, vector1, vector2)

        if vector1 is not None and vector2 is not None:
            cos = self.vectors.cosine_similarities(vector1, np.array([vector2]))[0]

        if np.isnan(cos):
            cos = 0
        if cos:
            cos = -cos if cos < 0 else cos
        return cos

    def similarity_matrix(self, vectors: List[np.ndarray]) -> np.ndarray:
        matrix = []
        for vector_1 in tqdm(vectors, total=len(vectors)):
            row = []
            for vector_2 in vectors:
                row.append(self.cosine(vector1=vector_1, vector2=vector_2))
            matrix.append(np.array(row))
        return np.array(matrix)

    @staticmethod
    def n_similarity(v1: Union[List[np.ndarray], np.ndarray], v2: Union[List[np.ndarray], np.ndarray]) -> np.ndarray:
        if isinstance(v1, list):
            v1 = np.array(v1)
        if isinstance(v2, list):
            v2 = np.array(v2)
        if not (len(v1) and len(v2)):
            raise ZeroDivisionError('At least one of the passed lists is empty.')
        return np.dot(matutils.unitvec(v1.mean(axis=0)), matutils.unitvec(v2.mean(axis=0)))

    def avg_embedding(self) -> np.ndarray:
        # def nan_checker(inp):
        #     if np.isnan(inp) or inp == np.nan:
        #         return 0
        #     else:
        #         return inp
        if self.oov_embedding is None:
            vecs = [self.vectors.get_vector(word) for word in self.vocab]
            # vecs = [np.array([nan_checker(ele) for ele in vec]) for vec in vecs]
            self.oov_embedding = sum(vecs) / len(vecs)
        return self.oov_embedding

    def get_concept_vector_old(self, concept) -> Union[np.ndarray, None]:
        if concept in self.vocab:
            # print("in", concept,  self.umls_mapper.umls_reverse_dict[concept])
            return self.vectors.get_vector(concept)
        else:
            if concept in self.umls_mapper.umls_reverse_dict:
                concept_vectors = []
                for candidate in self.umls_mapper.umls_reverse_dict[concept]:
                    candidate_tokens = candidate.split()
                    candidate_vectors = []
                    for token in candidate_tokens:
                        if token in self.vocab:
                            candidate_vectors.append(self.vectors.get_vector(token))

                    if len(candidate_vectors) == len(candidate_tokens):
                        candidate_vector = sum(candidate_vectors)
                        concept_vectors.append(candidate_vector)

                if len(concept_vectors) > 0:
                    concept_vector = sum(concept_vectors) / len(concept_vectors)
                    # print("not in", concept, self.umls_mapper.umls_reverse_dict[concept])
                    return concept_vector
            return None

    def get_concept_vector(self, concept) -> Union[np.ndarray, None]:
        try:
            return self.vectors.get_vector(concept)
        except KeyError:
            return self.avg_embedding()


class CategoryBenchmark(Benchmark):
    def __init__(self, embeddings: Tuple[gensim.models.KeyedVectors, str, str, str],
                 umls_mapper: UMLSMapper,
                 umls_evaluator: UMLSEvaluator):
        super().__init__(embeddings=embeddings, umls_mapper=umls_mapper, umls_evaluator=umls_evaluator)

        self.concept2category = {concept: category for concept, category in self.umls_evaluator.concept2category.items()
                                 if concept in self.vocab}

        self.category2concepts = revert_list_dict(self.concept2category)

    def evaluate(self) -> float:
        # score = 0.999
        score = self.all_categories_benchmark()
        return score

    def pairwise_cosine(self, concepts1, concepts2=None):
        if concepts2 is None:
            concepts2 = concepts1
            s = 0
            count = 0
            for i, concept1 in enumerate(concepts1):
                for j, concept2 in enumerate(concepts2):
                    if j > i:

                        cos_sim = self.cosine(concept1=concept1, concept2=concept2)
                        if cos_sim:
                            if cos_sim < 0:
                                cos_sim = -cos_sim
                            s += cos_sim
                            count += 1

            return s / count
        else:
            s = 0
            count = 0
            for i, concept1 in enumerate(concepts1):
                for j, concept2 in enumerate(concepts2):
                    cos_sim = self.cosine(concept1=concept1, concept2=concept2)
                    if cos_sim < 0:
                        cos_sim = -cos_sim
                    s += cos_sim
                    count += 1
            return s / count

    def category_benchmark(self, choosen_category):
        other_categories = self.category2concepts.keys()
        choosen_concepts = self.category2concepts[choosen_category]
        if len(choosen_concepts) <= 1:
            return 0, 0, 0
        p1 = self.pairwise_cosine(choosen_concepts)

        p2s = []
        for other_category in other_categories:
            if other_category == choosen_category:
                continue

            other_concepts = self.category2concepts[other_category]
            if len(choosen_concepts) == 0 or len(other_concepts) == 0:
                continue
            p2 = self.pairwise_cosine(choosen_concepts, other_concepts)
            p2s.append(p2)

        avg_p2 = sum(p2s) / len(p2s)
        return p1, avg_p2, p1 - avg_p2

    def all_categories_benchmark(self):

        distances = []
        print(self.category2concepts.keys())
        categories = tqdm(self.category2concepts.keys())
        for category in categories:
            within, out, distance = self.category_benchmark(category)
            distances.append(distance)
            categories.set_description(f"{category}: {within:.4f}|{out:.4f}|{distance:.4f}")
            categories.refresh()  # to show immediately the update

        benchmark_value = sum(distances) / len(distances)
        return benchmark_value


class SilhouetteCoefficient(Benchmark):
    def __init__(self, embeddings: Tuple[gensim.models.KeyedVectors, str, str, str],
                 umls_mapper: UMLSMapper,
                 umls_evaluator: UMLSEvaluator):
        super().__init__(embeddings=embeddings, umls_mapper=umls_mapper, umls_evaluator=umls_evaluator)
        self.concept2category = {concept: category for concept, category in self.umls_evaluator.concept2category.items()
                                 if concept in self.vocab}

        self.category2concepts = revert_list_dict(self.concept2category)

    def evaluate(self) -> float:
        score = self.silhouette_coefficient()
        return score

    def silhouette(self, term, category):
        def mean_between_distance(datapoint, same_cluster):
            sigma_ai = 0
            for reference_point in same_cluster:
                if datapoint == reference_point:
                    continue
                sigma_ai += self.cosine(concept1=datapoint, concept2=reference_point)

            return sigma_ai / (len(same_cluster) - 1)

        def smallest_mean_out_distance(datapoint, other_clusters):
            distances = []
            for other_cluster in other_clusters:
                sigma_bi = 0
                for other_reference_point in other_cluster:
                    sigma_bi += self.cosine(concept1=datapoint, concept2=other_reference_point)
                sigma_bi = sigma_bi / len(other_cluster)
                distances.append(sigma_bi)
            # return sum(distances)/len(distances) # alternative?
            return min(distances)

        cluster = self.category2concepts[category]
        other_cluster_names = set(self.category2concepts.keys()).difference(category)
        other_clusters_concepts = [self.category2concepts[category] for category in other_cluster_names]

        a_i = mean_between_distance(term, cluster)
        b_i = smallest_mean_out_distance(term, other_clusters_concepts)

        if a_i < b_i:
            s_i = 1 - a_i / b_i
        elif a_i == b_i:
            s_i = 0
        else:
            s_i = b_i / a_i - 1

        return s_i

    def silhouette_coefficient(self):
        s_is = []
        categories = tqdm(self.category2concepts.keys())
        for category in categories:
            category_concepts = self.category2concepts[category]
            if len(category_concepts) < 2:
                continue

            category_s_is = []
            for term in category_concepts:
                category_s_is.append(self.silhouette(term, category))

            mean_category_s_i = sum(category_s_is) / len(category_s_is)
            s_is.append(mean_category_s_i)
            categories.set_description(f"{category}: {mean_category_s_i:.4f}")
            categories.refresh()  # to show immediately the update
        return max(s_is)  # min, avg?


class EmbeddingSilhouetteCoefficient(Benchmark):
    def __init__(self, embeddings: Tuple[gensim.models.KeyedVectors, str, str, str],
                 umls_mapper: UMLSMapper,
                 umls_evaluator: UMLSEvaluator):
        super().__init__(embeddings=embeddings, umls_mapper=umls_mapper, umls_evaluator=umls_evaluator)
        self.concept2category = {concept: category for concept, category in self.umls_evaluator.concept2category.items()
                                 if concept in self.vocab}

        self.category2concepts = revert_list_dict(self.concept2category)

    def evaluate(self):
        score = self.silhouette_coefficient()
        return score

    def silhouette(self, term, category):
        def mean_between_distance(datapoint, same_cluster):
            sigma_ai = 0
            for reference_point in same_cluster:
                if datapoint == reference_point:
                    continue
                sigma_ai += self.cosine(concept1=datapoint, concept2=reference_point)

            return sigma_ai / (len(same_cluster) - 1)

        def smallest_mean_out_distance(datapoint, other_clusters):
            distances = []
            for other_cluster in other_clusters:
                sigma_bi = 0
                for other_reference_point in other_cluster:
                    sigma_bi += self.cosine(concept1=datapoint, concept2=other_reference_point)
                sigma_bi = sigma_bi / len(other_cluster)
                distances.append(sigma_bi)
            # return sum(distances)/len(distances) # alternative?
            return sum(distances) / len(distances)

        cluster = self.category2concepts[category]
        other_cluster_names = set(self.category2concepts.keys()).difference(category)
        other_clusters_concepts = [self.category2concepts[category] for category in other_cluster_names]

        a_i = mean_between_distance(term, cluster)
        b_i = smallest_mean_out_distance(term, other_clusters_concepts)

        if a_i < b_i:
            s_i = 1 - a_i / b_i
        elif a_i == b_i:
            s_i = 0
        else:
            s_i = b_i / a_i - 1

        return s_i

    def silhouette_coefficient(self):
        s_is = []
        categories = tqdm(self.category2concepts.keys())
        for category in categories:
            category_concepts = self.category2concepts[category]
            if len(category_concepts) < 2:
                continue

            category_s_is = []
            for term in category_concepts:
                category_s_is.append(self.silhouette(term, category))

            mean_category_s_i = sum(category_s_is) / len(category_s_is)
            s_is.append(mean_category_s_i)
            categories.set_description(f"{category}: {mean_category_s_i:.4f}")
            categories.refresh()  # to show immediately the update
        return sum(s_is) / len(s_is)  # min, max?


class Relation(Enum):
    MAY_TREAT = "may_treat"
    MAY_PREVENT = "may_prevent"


class ConceptualSimilarityChoi(Benchmark):
    def __init__(self, embeddings: Tuple[gensim.models.KeyedVectors, str, str, str],
                 umls_mapper: UMLSMapper,
                 umls_evaluator: UMLSEvaluator,
                 ndf_evaluator: NDFEvaluator):
        super().__init__(embeddings=embeddings, umls_mapper=umls_mapper)
        self.ndf_evaluator = ndf_evaluator
        self.umls_evaluator = umls_evaluator

        self.concept2category = {concept: category for concept, category in self.umls_evaluator.concept2category.items()
                                 if concept in self.vocab}

        self.category2concepts = revert_list_dict(self.concept2category)

    def evaluate(self):
        categories = ['Pharmacologic Substance',
                      'Disease or Syndrome',
                      'Neoplastic Process',
                      'Clinical Drug',
                      'Finding',
                      'Injury or Poisoning',
                      ]

        results = []
        for category in categories:
            category_result = self.mcsm(category)
            print(f'{self.dataset}|{self.preprocessing}|{self.algorithm} [{category}]: {category_result}')
            results.append(category_result)
        return sum(results)/len(results)

    def mcsm(self, category, k=40):
        # V: self.concept2category.keys()
        # T: category
        # k: k
        # V(t): v_t = self.category2concepts[category]
        # 1T: category_true

        def category_true(concept_neighbor, semantic_category):
            if semantic_category in self.concept2category[concept_neighbor]:
                return 1
            else:
                return 0

        v_t = self.category2concepts[category]
        if len(v_t) == 0:
            return 0

        sigma = 0

        for v in v_t:
            neighbors = self.vectors.most_similar(v, topn=k)
            for i in range(0, k):
                v_i = neighbors[i][0]

                if v_i in self.concept2category:
                    sigma += category_true(v_i, category) / math.log((i + 1) + 1, 2)
                # else: sigma += 0 (if neighbor not CUI ignore it)
        return sigma / len(v_t)


class MedicalRelatednessChoi(Benchmark, ABC):
    def __init__(self, embeddings: Tuple[gensim.models.KeyedVectors, str, str, str],
                 umls_mapper: UMLSMapper,
                 umls_evaluator: UMLSEvaluator,
                 ndf_evaluator: NDFEvaluator,
                 relation: Relation):
        super().__init__(embeddings=embeddings, umls_mapper=umls_mapper)
        self.ndf_evaluator = ndf_evaluator
        self.umls_evaluator = umls_evaluator

        self.concept2category = {concept: category for concept, category in self.umls_evaluator.concept2category.items()
                                 if concept in self.vocab}

        self.category2concepts = revert_list_dict(self.concept2category)
        self.relation = relation

    def evaluate(self):
        mean, max_value = self.run_mrm(relation=self.relation, sample=100)
        print(f'{self.relation} - mean: {mean}, max: {max_value}')

        return mean, max_value

    def mrm(self, relation_dictionary, v_star, seed_pair: Tuple[str, str] = None,
            k=40):
        # V: self.concept2category.keys()
        # R: relation_dictionary, relation_dictionary_reversed
        # k: k
        # V*: v_star
        # with the given relation
        # V(t): v_t = self.category2concepts[category]
        # 1R: relation_true
        # s: seed_pair
        def relation_true(selected_concept, concepts):
            for concept in concepts:
                related_terms = relation_dictionary.get(selected_concept)
                if related_terms and concept in related_terms:
                    return 1

            return 0

        s_difference = self.get_concept_vector(seed_pair[0]) - self.get_concept_vector(seed_pair[1])

        def compute_neighbor_relations(v, v_vector, s, vectors):
            neighbors = vectors.most_similar(positive=[v_vector - s], topn=k)
            neighbors = [tupl[0] for tupl in neighbors]
            return relation_true(selected_concept=v, concepts=neighbors)

        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores, backend="threading")(delayed(compute_neighbor_relations)
                                                                  (v_star_i, self.get_concept_vector(v_star_i),
                                                                   s_difference, self.vectors)
                                                                  for v_star_i in v_star)
        return sum(results) / len(v_star)

    def run_mrm(self, relation: Relation, sample: int = None):
        if relation == Relation.MAY_TREAT:
            relation_dict = self.ndf_evaluator.may_treat
            # relation_dict_reversed = self.ndf_evaluator.reverted_treat
        else:
            relation_dict = self.ndf_evaluator.may_prevent
            # relation_dict_reversed = self.ndf_evaluator.reverted_prevent

        v_star = set(relation_dict.keys())
        v_star = list(v_star)
        # v_star = [concept for concept in v_star if concept in self.vocab]

        seed_pairs = [(substance, disease) for substance, diseases in relation_dict.items() for disease in diseases]

        if sample:
            random.seed(42)
            seed_pairs = random.sample(seed_pairs, 100)

        tqdm_progress = tqdm(seed_pairs, total=len(seed_pairs))

        # results = Parallel(n_jobs=multiprocessing.cpu_count(), backend="threading") \
        #     (delayed(self.mrm)(relation_dict, relation_dict_reversed, v_star, seed_pair=seed_pair, k=40)
        #      for seed_pair in tqdm_progress)

        results = []
        for seed_pair in tqdm_progress:
            results.append(self.mrm(relation_dict, v_star, seed_pair=seed_pair, k=40))
            tqdm_progress.set_description(
                f'{seed_pair[0]}: [{results[-1]:.5f}, {sum(results) / len(results):.5f}, '
                f'{max(results):.5f}]')
            tqdm_progress.update()

        return sum(results) / len(results), max(results)


class MedicalRelatednessMayTreatChoi(MedicalRelatednessChoi):
    def __init__(self, embeddings: Tuple[gensim.models.KeyedVectors, str, str, str],
                 umls_mapper: UMLSMapper,
                 umls_evaluator: UMLSEvaluator,
                 ndf_evaluator: NDFEvaluator):
        super().__init__(embeddings=embeddings,
                         umls_mapper=umls_mapper, umls_evaluator=umls_evaluator, ndf_evaluator=ndf_evaluator,
                         relation=Relation.MAY_TREAT)


class MedicalRelatednessMayPreventChoi(MedicalRelatednessChoi):
    def __init__(self, embeddings: Tuple[gensim.models.KeyedVectors, str, str, str],
                 umls_mapper: UMLSMapper,
                 umls_evaluator: UMLSEvaluator,
                 ndf_evaluator: NDFEvaluator):
        super().__init__(embeddings=embeddings,
                         umls_mapper=umls_mapper, umls_evaluator=umls_evaluator, ndf_evaluator=ndf_evaluator,
                         relation=Relation.MAY_PREVENT)


# class ChoiBenchmark(Benchmark):
#     def __init__(self, embeddings: Tuple[gensim.models.KeyedVectors, str, str, str],
#                  umls_mapper: UMLSMapper,
#                  umls_evaluator: UMLSEvaluator,
#                  ndf_evaluator: NDFEvaluator):
#         super().__init__(embeddings=embeddings, umls_mapper=umls_mapper)
#         self.ndf_evaluator = ndf_evaluator
#         self.umls_evaluator = umls_evaluator
#
#         self.concept2category = {concept: category
#                                  for concept, category in self.umls_evaluator.concept2category.items()
#                                  if concept in self.vocab}
#
#         self.category2concepts = revert_list_dict(self.concept2category)
#
#     def evaluate(self):
#         categories = ['Pharmacologic Substance',
#                       'Disease or Syndrome',
#                       'Neoplastic Process',
#                       'Clinical Drug',
#                       'Finding',
#                       'Injury or Poisoning'
#                       ]
#
#         for category in categories:
#             print(f'{category}: {self.mcsm(category)}')
#
#         relations = [
#             Relation.MAY_TREAT,
#             Relation.MAY_PREVENT
#         ]
#         for relation in relations:
#             mean, max_value = self.run_mrm(relation=relation)
#             print(f'{relation} - mean: {mean}, max: {max_value}')
#
#     def mcsm(self, category, k=40):
#         def category_true(concept, semantic_category):
#             if semantic_category in self.concept2category[concept]:
#                 return 1
#             else:
#                 return 0
#
#         v_t = self.category2concepts[category]
#         if len(v_t) == 0:
#             return 0
#
#         sigma = 0
#         for v in v_t:
#             for i in range(0, k):
#                 neighbors = self.vectors.most_similar(v, topn=k)
#                 v_i = neighbors[i][0]
#                 # if word not in resource file ignore it
#                 if v_i in self.concept2category:
#                     sigma += category_true(v_i, category) / math.log((i + 1) + 1, 2)
#         return sigma / len(v_t)
#
#     def mrm(self, relation_dictionary, relation_dictionary_reversed, v_star, seed_pair: Tuple[str, str] = None, k=40):
#         def relation_true(selected_concept, concepts):
#             for concept in concepts:
#                 related_terms = relation_dictionary.get(selected_concept)
#                 if related_terms and concept in related_terms:
#                     return 1
#
#                 inverse_related_terms = relation_dictionary_reversed.get(selected_concept)
#                 if inverse_related_terms and concept in inverse_related_terms:
#                     return 1
#             return 0
#
#         def get_seed_pair():
#             for key, values in relation_dictionary.items():
#                 if key in v_star:
#                     for value in values:
#                         if value in v_star:
#                             return key, value
#
#             return self.umls_mapper.umls_dict["Cisplatin"], self.umls_mapper.umls_dict["Carboplatin"]
#
#         if seed_pair is None:
#             seed_pair = get_seed_pair()
#
#         s = self.vectors.get_vector(seed_pair[0]) - self.vectors.get_vector(seed_pair[1])
#
#         sigma = 0
#
#         for v in v_star:
#             neighbors = self.vectors.most_similar(positive=[self.vectors.get_vector(v) - s], topn=k)
#             neighbors = [tupl[0] for tupl in neighbors]
#
#             sigma += relation_true(selected_concept=v, concepts=neighbors)
#         return sigma / len(v_star)
#
#     def run_mrm(self, relation: Relation):
#
#         if relation == Relation.MAY_TREAT:
#             relation_dict = self.ndf_evaluator.may_treat
#             relation_dict_reversed = self.ndf_evaluator.reverted_treat
#         else:
#             relation_dict = self.ndf_evaluator.may_prevent
#             relation_dict_reversed = self.ndf_evaluator.reverted_prevent
#
#         v_star = set(relation_dict.keys())
#         v_star.update(relation_dict_reversed.keys())
#         v_star = [concept for concept in v_star if concept in self.vocab]
#
#         results = []
#         tqdm_progress = tqdm(relation_dict.items(), total=len(relation_dict.keys()))
#         for key, values in tqdm_progress:
#             if key in v_star:
#                 for value in values:
#                     if value in v_star:
#                         results.append(self.mrm(relation_dict, relation_dict_reversed, v_star,
#                                                 seed_pair=(key, value), k=40))
#                         tqdm_progress.set_description(f'{key}: [{results[-1]:.5f}, {sum(results) / len(results):.5f},'
#                                                       f' {max(results):.5f}]')
#                         tqdm_progress.update()
#
#         return sum(results) / len(results), max(results)


class HumanAssessmentTypes(Enum):
    RELATEDNESS = "relatedness"
    RELATEDNESS_CONT = "relatedness_cont"
    SIMILARITY_CONT = "similarity_cont"


class HumanAssessment(Benchmark):
    def __init__(self, embeddings: Tuple[gensim.models.KeyedVectors, str, str, str],
                 umls_mapper: UMLSMapper,
                 srs_evaluator: SRSEvaluator,
                 use_spearman: bool = True):
        super().__init__(embeddings=embeddings, umls_mapper=umls_mapper)
        self.srs_evaluator = srs_evaluator
        self.use_spearman = use_spearman

    def get_mae(self, human_assessment_dict):
        sigma = []
        for concept, other_concepts in human_assessment_dict.items():
            concept_vec = self.get_concept_vector(concept)
            if concept_vec is not None:
                for other_concept in other_concepts:
                    other_concept_vec = self.get_concept_vector(other_concept)
                    if other_concept_vec is not None:
                        distance = abs(human_assessment_dict[concept][other_concept]
                                       - self.cosine(vector1=concept_vec, vector2=other_concept_vec))
                        sigma.append(distance)

        print(f'found {len(sigma)} assessments in embeddings')
        return sum(sigma) / len(sigma)

    def get_spearman(self, human_assessment_dict):
        human_assessment_values = []
        cosine_values = []
        for concept, other_concepts in human_assessment_dict.items():
            concept_vec = self.get_concept_vector(concept)
            if concept_vec is not None:
                for other_concept in other_concepts:
                    other_concept_vec = self.get_concept_vector(other_concept)
                    if other_concept_vec is not None:
                        human_assessment_values.append(human_assessment_dict[concept][other_concept])
                        cosine_values.append(self.cosine(vector1=concept_vec, vector2=other_concept_vec))
        cor, p = spearmanr(human_assessment_values, cosine_values)
        return cor

    def human_assessments(self, human_assestment_type: HumanAssessmentTypes):
        if not self.use_spearman:
            if human_assestment_type == HumanAssessmentTypes.RELATEDNESS:
                return self.get_mae(self.srs_evaluator.human_relatedness)
            elif human_assestment_type == HumanAssessmentTypes.SIMILARITY_CONT:
                return self.get_mae(self.srs_evaluator.human_similarity_cont)
            else:
                return self.get_mae(self.srs_evaluator.human_relatedness_cont)
        else:
            if human_assestment_type == HumanAssessmentTypes.RELATEDNESS:
                return self.get_spearman(self.srs_evaluator.human_relatedness)
            elif human_assestment_type == HumanAssessmentTypes.SIMILARITY_CONT:
                return self.get_spearman(self.srs_evaluator.human_similarity_cont)
            else:
                return self.get_spearman(self.srs_evaluator.human_relatedness_cont)

    def evaluate(self) -> float:
        assessments = [
            HumanAssessmentTypes.SIMILARITY_CONT,
            # HumanAssessmentTypes.RELATEDNESS,
            # HumanAssessmentTypes.RELATEDNESS_CONT
        ]
        scores = []
        for assessment in assessments:
            score = self.human_assessments(assessment)
            scores.append(score)
            print(f'{assessment}: {score}')
        return sum(scores) / len(scores)


class SemanticTypeBeam(Benchmark):
    def __init__(self, embeddings: Tuple[gensim.models.KeyedVectors, str, str, str],
                 umls_mapper: UMLSMapper,
                 umls_evaluator: UMLSEvaluator):
        super().__init__(embeddings=embeddings, umls_mapper=umls_mapper, umls_evaluator=umls_evaluator)

    @staticmethod
    def bootstrap_samples(same_category_terms, different_category_terms, bootstraps: int = 10000, eps: float = 1e-6) \
            -> np.ndarray:
        def pmax(m1, m2):
            m2 = np.full(len(m1[0]), m2)
            resulting_arr = []
            for entry in m1:
                resulting_arr.append(np.array([max(*op_list) for op_list in zip(entry, m2)]))
            return np.array(resulting_arr)
        # a = 1:nrow(query_db)

        x_matrix = np.random.choice(list(same_category_terms.keys()), size=bootstraps, replace=True, p=None)
        # a = 1:nrow(results_db)
        y_matrix = np.random.choice(list(different_category_terms.keys()), size=bootstraps, replace=True, p=None)
        # print(X, Y)
        x_matrix = np.array([same_category_terms[key] for key in x_matrix])
        y_matrix = np.array([different_category_terms[key] for key in y_matrix])
        # print(X, Y)
        # query_rows = sample(x=1:nrow(query_db),size=bootstraps,replace=TRUE)
        # results_rows = sample(x=1:nrow(results_db),size=bootstraps,replace=TRUE)
        # X = query_db[query_rows,]
        # Y = results_db[results_rows,]

        # prevent division by zero by assigning at least eps of value
        # print(x_matrix[0])
        # print('X', x_matrix.shape)
        # print(y_matrix.shape)

        t1 = pmax(np.sqrt(x_matrix*x_matrix), eps)
        t2 = pmax(np.sqrt(y_matrix*y_matrix), eps)
        # print('pmax', t1.shape)

        # t1 = pmax(sqrt(apply(X, 1, crossprod)),eps)
        # t2 = pmax(sqrt(apply(Y, 1, crossprod)),eps)

        # ValueError: operands could not be broadcast together with shapes (10000,500) (10000,10000)
        x_matrix = x_matrix / t1
        y_matrix = y_matrix / t2
        print(x_matrix.shape, y_matrix.shape)
        # bootstrap_scores = rowSums(X * Y)
        bootstrap_scores = np.sum(x_matrix * y_matrix, axis=0)
        print(bootstrap_scores.shape)
        # print(bootstrap_scores)
        return bootstrap_scores

    # def real_beam(self):
    #     semantic_types = self.umls_evaluator.category2concepts.keys()
    #     all_concepts = set(self.umls_evaluator.concept2category.keys())
    #     power = 0
    #     for semantic_type in tqdm(semantic_types, total=len(semantic_types)):
    #         concepts_of_semantic_type = list(self.umls_evaluator.category2concepts[semantic_type])
    #         concepts_not_of_semantic_type = list(all_concepts.difference(concepts_of_semantic_type))
    #         # sim_scores = self.vectors.n_similarity(concepts_of_semantic_type, concepts_of_semantic_type)
    #         semantic_type_keyed_vectors = {concept: self.get_concept_vector(concept) for concept in concepts_of_semantic_type}
    #         not_semantic_type_keyed_vectors = {concept: self.get_concept_vector(concept) for concept in
    #                                      concepts_not_of_semantic_type}
    #
    #         semantic_type_keyed_vectors = {concept: vector for concept, vector in semantic_type_keyed_vectors.items() if
    #                                  vector is not None}
    #         not_semantic_type_keyed_vectors = {concept: vector for concept, vector in not_semantic_type_keyed_vectors.items() if
    #                                      vector is not None}
    #
    #         null_scores = self.bootstrap_samples(semantic_type_keyed_vectors, not_semantic_type_keyed_vectors)
    #         print('null_Scores shape', null_scores.shape)
    #         sig_threshold = np.quantile(null_scores, q=1 - constant.SIG_LEVEL)
    #         print(sig_threshold)
    #
    #         semantic_type_vecs = np.array([vector for concept_key, vector in semantic_type_keyed_vectors.items()])
    #         print(semantic_type_vecs.shape)
    #         sim_scores = self.n_similarity(semantic_type_vecs, semantic_type_vecs)
    #         # sim_scores = self.similarity_matrix(list(semantic_type_keyed_vectors.values()))
    #         print(sim_scores.shape)
    #         print(sim_scores)
    #         observed_scores = np.triu(sim_scores)
    #
    #         print(observed_scores)
    #
    #
    #
    #         num_positives = len(
    #             [observed_scores for observed_score in observed_scores if observed_score > sig_threshold])
    #         power = num_positives / len(observed_scores)
    #
    #     return power
    #
    # def own_beam(self):
    #     def sample(terms: Dict[str, np.ndarray], bootstraps: int = 10):
    #         sampled = np.random.choice(list(terms.keys()), size=bootstraps, replace=True, p=None)
    #         sampled_vecs = [terms[key] for key in sampled]
    #         return sampled_vecs
    #
    #     semantic_types = self.umls_evaluator.category2concepts.keys()
    #     all_concepts = set(self.umls_evaluator.concept2category.keys())
    #     semantic_sum = 0
    #     not_semantic_sum = 0
    #     tqdm_semantic_types = tqdm(semantic_types, total=len(semantic_types))
    #     for semantic_type in tqdm_semantic_types:
    #         concepts_of_semantic_type = list(self.umls_evaluator.category2concepts[semantic_type])
    #         concepts_not_of_semantic_type = list(all_concepts.difference(concepts_of_semantic_type))
    #
    #         semantic_type_vectors = {concept: self.get_concept_vector(concept) for concept in concepts_of_semantic_type}
    #         not_semantic_type_vectors = {concept: self.get_concept_vector(concept) for concept in
    #                                      concepts_not_of_semantic_type}
    #
    #         semantic_type_vectors = {concept: vector for concept, vector in semantic_type_vectors.items() if
    #                                  vector is not None}
    #         not_semantic_type_vectors = {concept: vector for concept, vector in not_semantic_type_vectors.items() if
    #                                      vector is not None}
    #
    #         if len(semantic_type_vectors) == 0:
    #             not_semantic_sum += 1
    #             continue
    #         sample_semantic = sample(semantic_type_vectors)
    #         sample_not_semantic = sample(not_semantic_type_vectors)
    #
    #         # print(np.triu(self.similarity_matrix(sample_not_semantic), k=1))
    #         semantic_matrix = self.similarity_matrix(sample_semantic)
    #         not_semantic_matrix = self.similarity_matrix(sample_not_semantic)
    #
    #         semantic_cosines = semantic_matrix[np.triu_indices(semantic_matrix.shape[0], k=1)]
    #         not_semantic_cosines = not_semantic_matrix[np.triu_indices(not_semantic_matrix.shape[0], k=1)]
    #         semantic_mean = semantic_cosines.mean()
    #         not_semantic_mean = not_semantic_cosines.mean()
    #         # fixme: adapt to real beam
    #         _, p_value = mannwhitneyu(semantic_cosines, not_semantic_cosines)
    #         if p_value < constant.SIG_LEVEL and semantic_mean > not_semantic_mean:
    #             semantic_sum += 1
    #         else:
    #             not_semantic_sum += 1
    #
    #         # semantic_mean - not_semantic_mean
    #         current_power = semantic_sum / (semantic_sum + not_semantic_sum)
    #         tqdm_semantic_types.set_description(f"{semantic_type}: {current_power:.4f}")
    #         tqdm_semantic_types.refresh()  # to show immediately the update
    #
    #     power = semantic_sum / (semantic_sum + not_semantic_sum)
    #     return power
    #
    # def improved_own_beam(self, sample_size=10000):
    #     def sample(elements: List[Tuple[str, str]], bootstraps: int = 10):
    #         random.seed(42)
    #         sampled = random.sample(elements, bootstraps)
    #         # sampled = np.random.choice(elements, size=bootstraps, replace=True, p=None)
    #         return sampled
    #
    #     def sample2(elements: List[Tuple[str, str]], other_elements: List[Tuple[str, str]], bootstraps: int = 10):
    #         random.seed(42)
    #         sampled = random.sample(elements, bootstraps)
    #         # sampled = np.random.choice(elements, size=bootstraps, replace=True, p=None)
    #         return sampled
    #
    #     def get_relations(concepts):
    #         relations = []
    #         for i, c1 in enumerate(concepts):
    #             for j, c2 in enumerate(concepts):
    #                 if j <= i:
    #                     continue
    #                 relations.append((c1, c2))
    #
    #         return relations
    #
    #     def pmax(l1, l2):
    #         if isinstance(l2, float):
    #             l2 = np.full(len(l1), l2)
    #         resulting_list = [max(*op_list) for op_list in zip(l1, l2)]
    #         if isinstance(l1, np.ndarray) and isinstance(l2, np.ndarray):
    #             return np.array(resulting_list)
    #         return resulting_list
    #
    #     def get_cosine_values_of_relations(tuple_list: List[Tuple[str, str]]) -> List[float]:
    #         return [self.cosine(vector1=self.get_concept_vector(rel[0]),
    #                             vector2=self.get_concept_vector(rel[1]))
    #                 for rel in tuple_list]
    #
    #     def null_distribution(rels, other_rels, eps=0.0000001):
    #         cosine_values_sample = np.array(get_cosine_values_of_relations(rels))
    #         other_cosine_values_sample = np.array(get_cosine_values_of_relations(other_rels))
    #         print('means', np.mean(cosine_values_sample), np.mean(other_cosine_values_sample))
    #         t1 = pmax(np.sqrt(cosine_values_sample * cosine_values_sample), eps)
    #         t2 = pmax(np.sqrt(other_cosine_values_sample * other_cosine_values_sample), eps)
    #         print(t1, t2)
    #         x = cosine_values_sample/t1
    #         y = other_cosine_values_sample/t2
    #         print(x)
    #         print(y)
    #         bootstrap_scores = np.sum(x * y, axis=0)
    #         print('bs', bootstrap_scores)
    #         t_score, p_value = ttest_ind(cosine_values_sample, other_cosine_values_sample, equal_var=False)
    #         print(t_score, p_value)
    #         # fixme: real null distribution, use other concepts
    #         return cosine_values_sample
    #
    #     def bootstrap(concepts, other_concepts):
    #         return [self.cosine(self.get_concept_vector(concept), self.get_concept_vector(other_concept))
    #          for concept, other_concept in zip(concepts, other_concepts)]
    #
    #
    #     total_positives = 0
    #     total_observed_scores = 0
    #     # categories = self.umls_evaluator.category2concepts.keys()
    #     categories = ['Pharmacologic Substance',
    #                   'Disease or Syndrome',
    #                   'Neoplastic Process',
    #                   'Clinical Drug',
    #                   'Finding',
    #                   'Injury or Poisoning',
    #                   ]
    #
    #     all_concepts = set(self.umls_evaluator.concept2category.keys())
    #     for category in tqdm(categories, total=len(categories)):
    #
    #         category_concepts = list(self.umls_evaluator.category2concepts[category])[:1000]
    #
    #         other_concepts = list(all_concepts.difference(category_concepts))[:1000]
    #         # print(len(category_concepts))
    #         # relations = {concepts in category}X{concepts in category}
    #         # fixme: outpfmemory
    #         relations = get_relations(category_concepts)
    #         # sample  two concepts c1, c2 of category
    #         sampled_relations = sample(relations, sample_size)
    #         sampled_other_relations = sample(get_relations(other_concepts), sample_size)
    #
    #         # print(cosine_values_sample)
    #         bootstrap()
    #         sig_threshold = np.quantile(null_distribution(sampled_relations, sampled_other_relations),
    #                                     1-constant.SIG_LEVEL)
    #         # print(sig_threshold)
    #         num_observed_scores = 0
    #         num_positives = 0
    #
    #         for relation in relations:
    #             observed_score = self.cosine(vector1=self.get_concept_vector(relation[0]),
    #                                          vector2=self.get_concept_vector(relation[1]))
    #
    #             if observed_score >= sig_threshold:
    #                 num_positives += 1
    #             num_observed_scores += 1
    #
    #         total_positives += num_positives
    #         total_observed_scores += num_observed_scores
    #         # category_power = num_positives / num_observed_scores
    #         # print(category_power)
    #
    #     power = total_positives / total_observed_scores
    #     return power

    def best_own_beam(self, sample_size=10000):
        def sample(elements: List, bootstraps: int = 10):
            if len(elements) < bootstraps:
                return elements
            random.seed(42)
            return random.sample(elements, bootstraps)

        def bootstrap(category_concepts, other_concepts):
            concept_sample = sample(category_concepts, sample_size)
            other_sample = sample(other_concepts, sample_size)

            bootstrap_values = [self.cosine(vector1=self.get_concept_vector(concept),
                                            vector2=self.get_concept_vector(other_concept))
                                for concept, other_concept in zip(concept_sample, other_sample)]

            threshold = np.quantile(bootstrap_values,
                                    1 - constant.SIG_LEVEL)

            return threshold

        total_positives = 0
        total_observed_scores = 0
        # categories = self.umls_evaluator.category2concepts.keys()
        categories = ['Pharmacologic Substance',
                      'Disease or Syndrome',
                      'Neoplastic Process',
                      'Clinical Drug',
                      'Finding',
                      'Injury or Poisoning',
                      ]

        all_concepts = set([concept for concept in self.umls_evaluator.concept2category.keys()
                            if concept in self.vocab])
        tqdm_bar = tqdm(categories, total=len(categories))
        for category in tqdm_bar:

            same_type_concepts = [concept for concept in self.umls_evaluator.category2concepts[category]
                                  if concept in self.vocab]
            other_type_concepts = list(all_concepts.difference(same_type_concepts))

            if len(same_type_concepts) == 0 or len(other_type_concepts) == 0:
                continue

            sig_threshold = bootstrap(same_type_concepts, other_type_concepts)
            tqdm_bar.set_description(f'{category}: t={sig_threshold:.4f}')
            tqdm_bar.update()
            num_observed_scores = 0
            num_positives = 0
            sampled_same_type_concepts = sample(same_type_concepts, 100)
            for i, first_category_concept in enumerate(sampled_same_type_concepts):
                for j, second_category_concept in enumerate(sampled_same_type_concepts):
                    if j <= i:
                        continue
                    observed_score = self.cosine(vector1=self.get_concept_vector(first_category_concept),
                                                 vector2=self.get_concept_vector(second_category_concept))

                    if observed_score >= sig_threshold:
                        num_positives += 1
                    num_observed_scores += 1

            total_positives += num_positives
            total_observed_scores += num_observed_scores
            # category_power = num_positives / num_observed_scores
            # print(category_power)

        power = total_positives / total_observed_scores
        return power

    def evaluate(self) -> float:
        power = self.best_own_beam()
        # power = self.own_beam()
        return power


class AbstractBeamBenchmark(Benchmark, ABC):
    def __init__(self, embeddings: Tuple[gensim.models.KeyedVectors, str, str, str],
                 umls_mapper: UMLSMapper,
                 umls_evaluator: UMLSEvaluator):
        super().__init__(embeddings=embeddings, umls_mapper=umls_mapper, umls_evaluator=umls_evaluator)

    @staticmethod
    def sample(elements: List, bootstraps: int = 10):
        if len(elements) < bootstraps:
            return elements
        random.seed(42)
        return random.sample(elements, bootstraps)

    def bootstrap(self, category_concepts: List[str], other_concepts: List[str], sample_size: int = 10000):

        concept_sample = self.sample(category_concepts, sample_size)
        other_sample = self.sample(other_concepts, sample_size)

        bootstrap_values = [self.cosine(vector1=self.get_concept_vector(concept),
                                        vector2=self.get_concept_vector(other_concept))
                            for concept, other_concept in zip(concept_sample, other_sample)]

        threshold = np.quantile(bootstrap_values, 1 - constant.SIG_LEVEL)

        return threshold

    @abstractmethod
    def calculate_power(self):
        pass

    def evaluate(self) -> float:
        return self.calculate_power()


class SemanticTypeBeamBenchmark(AbstractBeamBenchmark):
    def __init__(self, embeddings: Tuple[gensim.models.KeyedVectors, str, str, str],
                 umls_mapper: UMLSMapper,
                 umls_evaluator: UMLSEvaluator):
        super().__init__(embeddings=embeddings, umls_mapper=umls_mapper, umls_evaluator=umls_evaluator)

    def calculate_power(self):
        total_positives = 0
        total_observed_scores = 0
        # categories = self.umls_evaluator.category2concepts.keys()
        categories = ['Pharmacologic Substance',
                      'Disease or Syndrome',
                      'Neoplastic Process',
                      'Clinical Drug',
                      'Finding',
                      'Injury or Poisoning',
                      ]

        all_concepts = set([concept for concept in self.umls_evaluator.concept2category.keys()
                            if concept in self.vocab])
        tqdm_bar = tqdm(categories, total=len(categories))
        for category in tqdm_bar:
            same_type_concepts = [concept for concept in self.umls_evaluator.category2concepts[category]
                                  if concept in self.vocab]
            other_type_concepts = list(all_concepts.difference(same_type_concepts))

            if len(same_type_concepts) == 0 or len(other_type_concepts) == 0:
                continue

            sig_threshold = self.bootstrap(same_type_concepts, other_type_concepts, sample_size=10000)
            num_observed_scores = 0
            num_positives = 0
            sampled_same_type_concepts = self.sample(same_type_concepts, 100)
            for i, first_category_concept in enumerate(sampled_same_type_concepts):
                for j, second_category_concept in enumerate(sampled_same_type_concepts):
                    if j <= i:
                        continue
                    observed_score = self.cosine(vector1=self.get_concept_vector(first_category_concept),
                                                 vector2=self.get_concept_vector(second_category_concept))

                    if observed_score >= sig_threshold:
                        num_positives += 1
                    num_observed_scores += 1

            total_positives += num_positives
            total_observed_scores += num_observed_scores

        return total_positives / total_observed_scores


class NDFRTBeamBenchmark(AbstractBeamBenchmark):
    def __init__(self, embeddings: Tuple[gensim.models.KeyedVectors, str, str, str],
                 umls_mapper: UMLSMapper,
                 umls_evaluator: UMLSEvaluator,
                 ndf_evaluator: NDFEvaluator):
        super().__init__(embeddings=embeddings, umls_mapper=umls_mapper, umls_evaluator=umls_evaluator)
        self.ndf_evaluator = ndf_evaluator

    def get_concepts_of_semantic_types(self, semantic_types):
        concepts = set()
        for semantic_type in semantic_types:
            concepts.update([concept
                             for concept in self.umls_evaluator.category2concepts[semantic_type]
                             if concept in self.vocab])
        return list(concepts)

    def calculate_power(self):
        total_positives = 0
        total_observed_scores = 0
        treatment_conditions = []
        for treatment, conditions in self.ndf_evaluator.may_prevent.items():
            for condition in conditions:
                if treatment in self.vocab and condition in self.vocab:
                    treatment_conditions.append((treatment, condition))
                else:
                    total_observed_scores += 1

        for treatment, conditions in self.ndf_evaluator.may_treat.items():
            for condition in conditions:
                if treatment in self.vocab and condition in self.vocab:
                    treatment_conditions.append((treatment, condition))
                else:
                    total_observed_scores += 1

        tqdm_bar = tqdm(treatment_conditions, total=len(treatment_conditions))
        for treatment_condition in tqdm_bar:
            current_treatment, current_condition = treatment_condition
            if current_treatment in self.umls_evaluator.concept2category.keys() \
                    and current_condition in self.umls_evaluator.concept2category.keys():
                treatment_semantic_types = self.umls_evaluator.concept2category[current_treatment]
                condition_semantic_types = self.umls_evaluator.concept2category[current_condition]
            else:
                continue

            treatment_concepts = self.get_concepts_of_semantic_types(treatment_semantic_types)
            condition_concepts = self.get_concepts_of_semantic_types(condition_semantic_types)

            if len(treatment_concepts) == 0 or len(condition_concepts) == 0:
                continue

            sig_threshold = self.bootstrap(treatment_concepts, condition_concepts, sample_size=10000)

            num_observed_scores = 0
            num_positives = 0

            observed_score = self.cosine(vector1=self.get_concept_vector(current_treatment),
                                         vector2=self.get_concept_vector(current_condition))
            if observed_score >= sig_threshold:
                num_positives += 1
            num_observed_scores += 1

            total_positives += num_positives
            total_observed_scores += num_observed_scores

        return total_positives / total_observed_scores


