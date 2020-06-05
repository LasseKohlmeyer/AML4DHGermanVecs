import math
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import Tuple, Dict, Set, Iterable, List, Union
import pandas as pd
import gensim
from scipy.stats import mannwhitneyu
from gensim import matutils
from numpy import dot
from tqdm import tqdm
import numpy as np

import constant
from UMLS import UMLSMapper, UMLSEvaluator
from embeddings import Embeddings
from evaluation_resource import NDFEvaluator, SRSEvaluator
from fasttext import load_model as load_fasttext_model
# from gensim.models.wrappers import FastText
from gensim.models.fasttext import load_facebook_model
from gensim.models.word2vec import Word2Vec
import gensim.models.keyedvectors as word2vec


def revert_list_dict(dictionary: Dict[str, Set[str]], filter_collection: Iterable = None) -> Dict[str, Set[str]]:
    reverted_dictionary = defaultdict(set)
    for key, values in dictionary.items():
        for value in values:
            if not filter_collection or value in filter_collection:
                reverted_dictionary[value].add(key)
    return reverted_dictionary




class AbstractBenchmark(ABC):
    def __init__(self, embeddings: Tuple[Union[gensim.models.KeyedVectors], str, str],
                 umls_mapper: UMLSMapper,
                 umls_evaluator: UMLSEvaluator = None):
        self.vectors, self.dataset, self.algorithm = embeddings
        try:
            self.vocab = self.vectors.vocab
        except AttributeError:
            self.vocab = self.vectors.vocabulary
        self.umls_mapper = umls_mapper
        if umls_evaluator:
            self.umls_evaluator = umls_evaluator

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

        if cos:
            cos = -cos if cos < 0 else cos
        return cos

    def similarity_matrix(self, vectors: List[np.ndarray]) -> np.ndarray:
        matrix = []
        for vector_1 in vectors:
            row = []
            for vector_2 in vectors:
                row.append(self.cosine(vector1=vector_1, vector2=vector_2))
            matrix.append(np.array(row))
        return np.array(matrix)


    def n_similarity(self, v1: List[np.ndarray], v2: List[np.ndarray]) -> np.ndarray:
        if not(len(v1) and len(v2)):
            raise ZeroDivisionError('At least one of the passed list is empty.')
        return np.dot(matutils.unitvec(np.array(v1).mean(axis=0)), matutils.unitvec(np.array(v2).mean(axis=0)))

    def get_concept_vector(self, concept) -> Union[np.ndarray, None]:
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


class CategoryBenchmark(AbstractBenchmark):
    def __init__(self, embeddings: Tuple[gensim.models.KeyedVectors, str, str],
                 umls_mapper: UMLSMapper,
                 umls_evaluator: UMLSEvaluator):
        super().__init__(embeddings=embeddings, umls_mapper=umls_mapper, umls_evaluator=umls_evaluator)

        self.concept2category = {concept: category for concept, category in self.umls_evaluator.concept2category.items()
                                 if concept in self.vocab}

        self.category2concepts = revert_list_dict(self.concept2category)

    def evaluate(self) -> float:
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


class SilhouetteCoefficient(AbstractBenchmark):
    def __init__(self, embeddings: Tuple[gensim.models.KeyedVectors, str, str],
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


class EmbeddingSilhouetteCoefficient(AbstractBenchmark):
    def __init__(self, embeddings: Tuple[gensim.models.KeyedVectors, str, str],
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
            return sum(distances)/len(distances)

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
        return sum(s_is)/len(s_is)  # min, max?


class Relation(Enum):
    MAY_TREAT = "may_treat"
    MAY_PREVENT = "may_prevent"


class ChoiBenchmark(AbstractBenchmark):
    def __init__(self, embeddings: Tuple[gensim.models.KeyedVectors, str, str],
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
                      'Injury or Poisoning'
                      ]

        for category in categories:
            print(f'{category}: {self.mcsm(category)}')

        relations = [
            Relation.MAY_TREAT,
            Relation.MAY_PREVENT
        ]
        for relation in relations:
            mean, max_value = self.run_mrm(relation=relation)
            print(f'{relation} - mean: {mean}, max: {max_value}')

    def mcsm(self, category, k=40):
        def category_true(concept, semantic_category):
            if semantic_category in self.concept2category[concept]:
                return 1
            else:
                return 0

        v_t = self.category2concepts[category]
        if len(v_t) == 0:
            return 0

        sigma = 0
        for v in v_t:
            for i in range(0, k):
                neighbors = self.vectors.most_similar(v, topn=k)
                v_i = neighbors[i][0]
                # if word not in resource file ignore it
                if v_i in self.concept2category:
                    sigma += category_true(v_i, category) / math.log((i + 1) + 1, 2)
        return sigma / len(v_t)

    def mrm(self, relation_dictionary, relation_dictionary_reversed, v_star, seed_pair: Tuple[str, str] = None, k=40):
        def relation_true(selected_concept, concepts):
            for concept in concepts:
                related_terms = relation_dictionary.get(selected_concept)
                if related_terms and concept in related_terms:
                    return 1

                inverse_related_terms = relation_dictionary_reversed.get(selected_concept)
                if inverse_related_terms and concept in inverse_related_terms:
                    return 1
            return 0

        def get_seed_pair():
            for key, values in relation_dictionary.items():
                if key in v_star:
                    for value in values:
                        if value in v_star:
                            return key, value

            return self.umls_mapper.umls_dict["Cisplatin"], self.umls_mapper.umls_dict["Carboplatin"]

        if seed_pair is None:
            seed_pair = get_seed_pair()

        s = self.vectors.get_vector(seed_pair[0]) - self.vectors.get_vector(seed_pair[1])

        sigma = 0

        for v in v_star:
            neighbors = self.vectors.most_similar(positive=[self.vectors.get_vector(v) - s], topn=k)
            neighbors = [tupl[0] for tupl in neighbors]

            sigma += relation_true(selected_concept=v, concepts=neighbors)
        return sigma / len(v_star)

    def run_mrm(self, relation: Relation):

        if relation == Relation.MAY_TREAT:
            relation_dict = self.ndf_evaluator.may_treat
            relation_dict_reversed = self.ndf_evaluator.reverted_treat
        else:
            relation_dict = self.ndf_evaluator.may_prevent
            relation_dict_reversed = self.ndf_evaluator.reverted_prevent

        v_star = set(relation_dict.keys())
        v_star.update(relation_dict_reversed.keys())
        v_star = [concept for concept in v_star if concept in self.vocab]

        results = []
        tqdm_progress = tqdm(relation_dict.items(), total=len(relation_dict.keys()))
        for key, values in tqdm_progress:
            if key in v_star:
                for value in values:
                    if value in v_star:
                        results.append(self.mrm(relation_dict, relation_dict_reversed, v_star,
                                                seed_pair=(key, value), k=40))
                        tqdm_progress.set_description(f'{key}: [{results[-1]:.5f}, {sum(results) / len(results):.5f}, '
                                                      f'{max(results):.5f}]')
                        tqdm_progress.update()

        return sum(results)/len(results), max(results)


class HumanAssessmentTypes(Enum):
    RELATEDNESS = "relatedness"
    RELATEDNESS_CONT = "relatedness_cont"
    SIMILARITY_CONT = "similarity_cont"


class HumanAssessment(AbstractBenchmark):
    def __init__(self,  embeddings: Tuple[gensim.models.KeyedVectors, str, str],
                 umls_mapper: UMLSMapper,
                 srs_evaluator: SRSEvaluator):
        super().__init__(embeddings=embeddings, umls_mapper=umls_mapper)
        self.srs_evaluator = srs_evaluator

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
                        # .similarity(concept, other_concept))
                        sigma.append(distance)
            #         else:
            #             sigma.append(1)
            # else:
            #     sigma.append(1)

        print(f'found {len(sigma)} assessments in embeddings')
        return sum(sigma)/len(sigma)

    def human_assessments(self, type: HumanAssessmentTypes):
        if type == HumanAssessmentTypes.RELATEDNESS:
            return self.get_mae(self.srs_evaluator.human_relatedness)
        elif type == HumanAssessmentTypes.SIMILARITY_CONT:
            return self.get_mae(self.srs_evaluator.human_similarity_cont)
        else:
            return self.get_mae(self.srs_evaluator.human_relatedness_cont)

    def evaluate(self) -> float:
        assessments = [HumanAssessmentTypes.SIMILARITY_CONT, HumanAssessmentTypes.RELATEDNESS, HumanAssessmentTypes.RELATEDNESS_CONT]
        scores = []
        for assessment in assessments:
            score = self.human_assessments(assessment)
            scores.append(score)
            print(f'{assessment}: {score}')
        return sum(scores)/len(scores)


class SemanticTypeBeam(AbstractBenchmark):
    def __init__(self,  embeddings: Tuple[gensim.models.KeyedVectors, str, str],
                 umls_mapper: UMLSMapper,
                 umls_evaluator: UMLSEvaluator):
        super().__init__(embeddings=embeddings, umls_mapper=umls_mapper, umls_evaluator=umls_evaluator)

    def bootstrap_samples(self, same_category_terms, different_category_terms, bootstraps: int = 10000, eps: float = 1e-6) -> np.ndarray:
        # a = 1:nrow(query_db)

        X = np.random.choice(list(same_category_terms.keys()), size=bootstraps, replace=True, p=None)
        # a = 1:nrow(results_db)
        Y = np.random.choice(list(different_category_terms.keys()), size=bootstraps, replace=True, p=None)
        # print(X, Y)
        X = np.array([same_category_terms[key] for key in X])
        Y = np.array([different_category_terms[key] for key in Y])
        # print(X, Y)
        # query_rows = sample(x=1:nrow(query_db),size=bootstraps,replace=TRUE)
        # results_rows = sample(x=1:nrow(results_db),size=bootstraps,replace=TRUE)
        # X = query_db[query_rows,]
        # Y = results_db[results_rows,]
        print(X.shape, Y.shape)
        # prevent division by zero by assigning at least eps of value
        t1 = np.maximum(np.sqrt(np.cross(X, X, axisa=0, axisb=0)), eps)
        t2 = np.maximum(np.sqrt(np.cross(X, X, axisa=0, axisb=0)), eps)

        # t1 = pmax(sqrt(apply(X, 1, crossprod)),eps)
        # t2 = pmax(sqrt(apply(Y, 1, crossprod)),eps)

        X = X/t1
        Y = Y/t2

        # bootstrap_scores = rowSums(X * Y)
        bootstrap_scores = np.sum(X * Y, axis=0)
        print(bootstrap_scores)
        return bootstrap_scores

    def real_beam(self):
        semantic_types = self.umls_evaluator.category2concepts.keys()
        all_concepts = set(self.umls_evaluator.concept2category.keys())
        power = 0
        for semantic_type in tqdm(semantic_types, total=len(semantic_types)):
            concepts_of_semantic_type = list(self.umls_evaluator.category2concepts[semantic_type])
            concepts_not_of_semantic_type = list(all_concepts.difference(concepts_of_semantic_type))
            # sim_scores = self.vectors.n_similarity(concepts_of_semantic_type, concepts_of_semantic_type)
            semantic_type_vectors = {concept: self.get_concept_vector(concept) for concept in concepts_of_semantic_type}
            not_semantic_type_vectors = {concept: self.get_concept_vector(concept) for concept in
                                         concepts_not_of_semantic_type}

            semantic_type_vectors = {concept: vector for concept, vector in semantic_type_vectors.items() if
                                     vector is not None}
            not_semantic_type_vectors = {concept: vector for concept, vector in not_semantic_type_vectors.items() if
                                         vector is not None}

            # sim_scores = self.n_similarity(semantic_type_vectors, not_semantic_type_vectors)
            sim_scores = self.similarity_matrix(list(semantic_type_vectors.values()))
            print(sim_scores.shape)
            observed_scores = np.triu(sim_scores)

            print(observed_scores)

            null_scores = self.bootstrap_samples(semantic_type_vectors, not_semantic_type_vectors)

            sig_threshold = np.quantile(null_scores, p=1 - constant.SIG_LEVEL)

            num_positives = len(
                [observed_scores for observed_score in observed_scores if observed_score > sig_threshold])
            power = num_positives / len(observed_scores)

        return power

    def own_beam(self):
        def sample(terms: Dict[str, np.ndarray], bootstraps: int = 10):
            sampled = np.random.choice(list(terms.keys()), size=bootstraps, replace=True, p=None)
            sampled_vecs = [terms[key] for key in sampled]
            return sampled_vecs

        semantic_types = self.umls_evaluator.category2concepts.keys()
        all_concepts = set(self.umls_evaluator.concept2category.keys())
        power = 0
        semantic_sum = 0
        not_semantic_sum = 0
        tqdm_semantic_types = tqdm(semantic_types, total=len(semantic_types))
        for semantic_type in tqdm_semantic_types:
            concepts_of_semantic_type = list(self.umls_evaluator.category2concepts[semantic_type])
            concepts_not_of_semantic_type = list(all_concepts.difference(concepts_of_semantic_type))

            semantic_type_vectors = {concept: self.get_concept_vector(concept) for concept in concepts_of_semantic_type}
            not_semantic_type_vectors = {concept: self.get_concept_vector(concept) for concept in
                                         concepts_not_of_semantic_type}

            semantic_type_vectors = {concept: vector for concept, vector in semantic_type_vectors.items() if
                                     vector is not None}
            not_semantic_type_vectors = {concept: vector for concept, vector in not_semantic_type_vectors.items() if
                                         vector is not None}

            if len(semantic_type_vectors) == 0:
                not_semantic_sum += 1
                continue
            sample_semantic = sample(semantic_type_vectors)
            sample_not_semantic = sample(not_semantic_type_vectors)

            # print(np.triu(self.similarity_matrix(sample_not_semantic), k=1))
            semantic_matrix = self.similarity_matrix(sample_semantic)
            not_semantic_matrix = self.similarity_matrix(sample_not_semantic)

            semantic_cosines = semantic_matrix[np.triu_indices(semantic_matrix.shape[0], k=1)]
            not_semantic_cosines = not_semantic_matrix[np.triu_indices(not_semantic_matrix.shape[0], k=1)]
            semantic_mean = semantic_cosines.mean()
            not_semantic_mean = not_semantic_cosines.mean()
            # fixme: adapt to real beam
            _, p_value = mannwhitneyu(semantic_cosines, not_semantic_cosines)
            if p_value < constant.SIG_LEVEL and semantic_mean > not_semantic_mean:
                semantic_sum += 1
            else:
                not_semantic_sum += 1

            # semantic_mean - not_semantic_mean
            current_power = semantic_sum / (semantic_sum + not_semantic_sum)
            tqdm_semantic_types.set_description(f"{semantic_type}: {current_power:.4f}")
            tqdm_semantic_types.refresh()  # to show immediately the update

        power = semantic_sum / (semantic_sum + not_semantic_sum)
        return power

    def evaluate(self) -> float:
        # power = self.real_beam()
        power = self.own_beam()
        return power


class Evaluation:
    def __init__(self, embeddings: List[Tuple[gensim.models.KeyedVectors, str, str]],
                 umls_mapper: UMLSMapper,
                 umls_evaluator: UMLSEvaluator,
                 ndf_evaluator: NDFEvaluator,
                 srs_evaluator: SRSEvaluator, benchmarkNames=List[str]):

        self.benchmarks = []
        for embedding in embeddings:
            # self.benchmarks.append(CategoryBenchmark(embedding, umls_mapper, umls_evaluator))
            # self.benchmarks.append(SemanticTypeBeam(embedding, umls_mapper, umls_evaluator))
            # self.benchmarks.append(SilhouetteCoefficient(embedding, umls_mapper, umls_evaluator))
            # self.benchmarks.append(ChoiBenchmark(embedding, umls_mapper, umls_evaluator, ndf_evaluator))
            self.benchmarks.append(HumanAssessment(embedding, umls_mapper, srs_evaluator))

    def evaluate(self):
        tuples = []
        for benchmark in self.benchmarks:
            print(benchmark.__class__.__name__, benchmark.dataset, benchmark.algorithm)
            score = benchmark.evaluate()
            number_concepts = len(set(benchmark.umls_mapper.umls_reverse_dict.keys()).intersection(set(benchmark.vocab)))
            tuples.append((benchmark.dataset, benchmark.algorithm, benchmark.__class__.__name__, score,number_concepts))

        df = pd.DataFrame(tuples, columns=['Data set', 'Algorithm', 'Benchmark', 'Score', '# Concepts'])
        print(df)
        df.to_csv('benchmark_results1.csv', index=False, encoding="utf-8")
        used_benchmarks_dict = defaultdict(list)
        for i, row in df.iterrows():
            used_benchmarks_dict["Data set"].append(row["Data set"])
            used_benchmarks_dict["Algorithm"].append(row["Algorithm"])
            used_benchmarks_dict[row["Benchmark"]].append(row["Score"])
            used_benchmarks_dict["# Concepts"].append(row["# Concepts"])

        df_table = pd.DataFrame.from_dict(used_benchmarks_dict)
        print(df_table)
        df.to_csv('benchmark_results2.csv', index=False, encoding="utf-8")

    # def analogies(vectors, start, minus, plus, umls: UMLSMapper):
#     if umls:
#         return vectors.most_similar(positive=[umls.umls_dict[start], umls.umls_dict[plus]],
#                                     negative=[umls.umls_dict[minus]])
#     else:
#         return vectors.most_similar(positive=[start, plus], negative=[minus])
#
#
# def similarities(vectors, word, umls):
#     if umls:
#         return vectors.most_similar(umls.umls_dict[word])
#     else:
#         return vectors.most_similar(word)


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
            addable_vectors.append(sum(concept_vec)/len(concept_vec))
    vectors.add(addable_concepts, addable_vectors)
    print(len(addable_concepts))
    return vectors


def main():
    umls_mapper = UMLSMapper(from_dir='E:/AML4DH-DATA/UMLS')

    ggponc_vecs = (Embeddings.load(path="data/no_prep_vecs_test_all.kv"), "GGPONC", "word2vec")
    ggponc_vecs_fasttext = (Embeddings.load(path="data/GGPONC_fastText_all.kv"), "GGPONC", "fastText")
    # https://devmount.github.io/GermanWordEmbeddings/
    # pretrained_wiki_news_vecs = (word2vec.KeyedVectors.load_word2vec_format('E:/german.model', binary=True),
    #                              "Wikipedia + News 2015", "word2vec")
    #
    # pretrained_wiki_news_vecs = (assign_concepts_to_vecs(pretrained_wiki_news_vecs[0], umls_mapper),
    #                              "Wikipedia + News 2015", "word2vec")

    news_vecs = (Embeddings.load(path="data/60K_news_all.kv"), "News 60K", "word2vec")
    news_vecs_big = (Embeddings.load(path="data/500K_news_all.kv"), "News 500K", "word2vec")
    news_vecs_big_3M = (Embeddings.load(path="data/3M_news_all.kv"), "News 3M", "word2vec")
    news_vecs_fasttext = (Embeddings.load(path="data/60K_news_all.kv"), "News 60K", "fastText")


    # fasttext_model = load_fasttext_model('E://cc.de.300.bin')
    # fasttext_vecs = (load_facebook_model('E:/cc.de.300.bin'), "common crawl", "fastText")
    # print(fasttext_vecs[0].vocabulary.)
    umls_evaluator = UMLSEvaluator(from_dir='E:/AML4DH-DATA/UMLS')

    ndf_evaluator = NDFEvaluator(from_dir='E:/AML4DH-DATA/NDF')

    srs_evaluator = SRSEvaluator(from_dir="E:/AML4DH-DATA/SRS")

    print(srs_evaluator.human_relatedness)

    # for c, v in ggponc_vecs.most_similar("Cisplatin"):
    #     print(umls_mapper.un_umls(c), v)
    #
    #
    # # for c, v in ggponc_vecs.most_similar(umls_mapper.umls_dict["Cisplatin"]):
    # #     print(umls_mapper.un_umls(c), v)
    #
    # for c, v in analogies(ggponc_vecs, "Asthma", "Lunge", "Herz", umls=umls_mapper):
    #     print(umls_mapper.un_umls(c), v)
    #
    # for c, v in similarities(ggponc_vecs, "Hepatitis", umls=umls_mapper):
    #     print(umls_mapper.un_umls(c), v)
    #
    # for c, v in similarities(ggponc_vecs, "Cisplatin", umls=umls_mapper):
    #     print(umls_mapper.un_umls(c), v)

    # print([(umls_mapper.un_umls(c), Embedding(umls_mapper.un_umls(c), ggponc_vecs[c])) for c in ggponc_vecs.vocab])

    # benchmark = CategoryBenchmark(ggponc_vecs, umls_mapper, evaluator)
    # benchmark.evaluate()

    evaluation = Evaluation([news_vecs, news_vecs_big, news_vecs_fasttext, news_vecs_big_3M,
                             ggponc_vecs, ggponc_vecs_fasttext],
                            umls_mapper, umls_evaluator, ndf_evaluator, srs_evaluator)
    evaluation.evaluate()

    # benchmark.category_benchmark("Nucleotide Sequence")

    # emb = EmbeddingSet( {umls_mapper.un_umls(c, single_return=True): Embedding(umls_mapper.un_umls(c,
    # single_return=True), ggponc_vecs[c]) for c in ggponc_vecs.vocab}) # emb = EmbeddingSet({c: Embedding(c, ggponc_vecs[c]) for c in
    # ggponc_vecs.vocab})
    #
    # emb.plot_interactive("Fibroblasten", "Fremdkörper")

    # replace multi words


if __name__ == "__main__":
    main()
