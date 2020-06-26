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
from UMLS import UMLSMapper, UMLSEvaluator, MRRELEvaluator
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
        tqdm_bar = tqdm(categories)
        for category in tqdm_bar:
            category_result = self.mcsm(category)
            # print(f'{self.dataset}|{self.preprocessing}|{self.algorithm} [{category}]: {category_result}')
            results.append(category_result)
            tqdm_bar.set_description(f"Conceptual Similarity Choi ({self.dataset}|{self.algorithm}|"
                                     f"{self.preprocessing}): "
                                     f"{category_result:.4f}")
            tqdm_bar.update()
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
        # print(f'{self.relation} - mean: {mean}, max: {max_value}')
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
                f'Medical Relatedness {self.relation} Choi ({self.dataset}|{self.algorithm}|{self.preprocessing}): '
                f'{sum(results) / len(results):.5f} mean, '
                f'{max(results):.5f} max')
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


class HumanAssessmentTypes(Enum):
    RELATEDNESS = "relatedness"
    RELATEDNESS_CONT = "relatedness_cont"
    SIMILARITY_CONT = "similarity_cont"
    MAYOSRS = "MayoSRS"


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
        tqdm_bar = tqdm(human_assessment_dict.items())
        for concept, other_concepts in tqdm_bar:
            concept_vec = self.get_concept_vector(concept)
            if concept_vec is not None:
                for other_concept in other_concepts:
                    other_concept_vec = self.get_concept_vector(other_concept)
                    if other_concept_vec is not None:
                        distance = abs(human_assessment_dict[concept][other_concept]
                                       - self.cosine(vector1=concept_vec, vector2=other_concept_vec))
                        sigma.append(distance)
                        tqdm_bar.set_description(f"Human Assessment ({self.dataset}|{self.algorithm}|"
                                                 f"{self.preprocessing}): "
                                                 f"{sum(sigma) / len(sigma):.4f}")
                        tqdm_bar.update()

        # print(f'found {len(sigma)} assessments in embeddings')
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
            HumanAssessmentTypes.RELATEDNESS,
            HumanAssessmentTypes.RELATEDNESS_CONT,
            HumanAssessmentTypes.MAYOSRS
        ]
        scores = []
        tqdm_bar = tqdm(assessments)
        for assessment in tqdm_bar:
            score = self.human_assessments(assessment)
            scores.append(score)
            tqdm_bar.set_description(f"Human Assessment Beam ({self.dataset}|{self.algorithm}|{self.preprocessing}): "
                                     f"{score:.4f}")
            tqdm_bar.update()

        return sum(scores) / len(scores)


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


class SemanticTypeBeam(AbstractBeamBenchmark):
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
            sampled_same_type_concepts = self.sample(same_type_concepts, 2000)
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
            tqdm_bar.set_description(f"Semantic Type Beam ({self.dataset}|{self.algorithm}|{self.preprocessing}): "
                                     f"{sig_threshold:.4f} threshold, "
                                     f"{(total_positives / total_observed_scores):.4f} score")
            tqdm_bar.update()

        return total_positives / total_observed_scores


class NDFRTBeam(AbstractBeamBenchmark):
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
            tqdm_bar.set_description(f"NDFRT Beam ({self.dataset}|{self.algorithm}|{self.preprocessing}): "
                                     f"{sig_threshold:.4f} threshold, "
                                     f"{(total_positives / total_observed_scores):.4f} score")
            tqdm_bar.update()

        return total_positives / total_observed_scores


class CausalityBeam(AbstractBeamBenchmark):
    def __init__(self, embeddings: Tuple[gensim.models.KeyedVectors, str, str, str],
                 umls_mapper: UMLSMapper,
                 umls_evaluator: UMLSEvaluator,
                 mrrelevaluator: MRRELEvaluator):
        super().__init__(embeddings=embeddings, umls_mapper=umls_mapper, umls_evaluator=umls_evaluator)
        self.mrrelevaluator = mrrelevaluator

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
        causative_relations = []
        for cause, effects in self.mrrelevaluator.mrrel.items():
            for effect in effects:
                if cause in self.vocab and effect in self.vocab:
                    causative_relations.append((cause, effect))
                # else:
                #     total_observed_scores += 1

        tqdm_bar = tqdm(causative_relations, total=len(causative_relations))
        for treatment_condition in tqdm_bar:
            current_cause, current_effect = treatment_condition
            if current_cause in self.umls_evaluator.concept2category.keys() \
                    and current_effect in self.umls_evaluator.concept2category.keys():
                cause_semantic_types = self.umls_evaluator.concept2category[current_cause]
                effect_semantic_types = self.umls_evaluator.concept2category[current_effect]
            else:
                continue

            cause_concepts = self.get_concepts_of_semantic_types(cause_semantic_types)
            effect_concepts = self.get_concepts_of_semantic_types(effect_semantic_types)

            if len(cause_concepts) == 0 or len(effect_concepts) == 0:
                continue

            sig_threshold = self.bootstrap(cause_concepts, effect_concepts, sample_size=10000)

            num_observed_scores = 0
            num_positives = 0

            observed_score = self.cosine(vector1=self.get_concept_vector(current_cause),
                                         vector2=self.get_concept_vector(current_effect))
            if observed_score >= sig_threshold:
                num_positives += 1
            num_observed_scores += 1

            total_positives += num_positives
            total_observed_scores += num_observed_scores
            tqdm_bar.set_description(f"Causality Beam ({self.dataset}|{self.algorithm}|{self.preprocessing}): "
                                     f"{sig_threshold:.4f} threshold, "
                                     f"{(total_positives / total_observed_scores):.4f} score")
            tqdm_bar.update()

        return total_positives / total_observed_scores


