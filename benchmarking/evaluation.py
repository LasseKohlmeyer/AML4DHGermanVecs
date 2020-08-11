import os
from collections import defaultdict
from typing import Tuple, List

import gensim

from resource.UMLS import UMLSMapper
from benchmarking.benchmarks import Benchmark
from resource.other_resources import Evaluator
import gc
import pandas as pd
import numpy as np

from vectorization.embeddings import Embedding


class Evaluation:
    def __init__(self, embeddings: List[Embedding],
                 umls_mapper: UMLSMapper,
                 evaluators: List[Evaluator],
                 benchmark_classes=List[Benchmark],
                 extend: bool = False):
        self.benchmark_classes = benchmark_classes
        self.embeddings = embeddings
        self.evaluators = evaluators
        self.umls_mapper = umls_mapper
        #
        # self.benchmarks = [benchmark(embedding, umls_mapper, evaluators)
        #                    for embedding in embeddings
        #                    for benchmark in benchmark_classes]

        self.evaluate()

    @staticmethod
    def build_paper_table(cache_df: pd.DataFrame, out_path: str):
        used_benchmarks_dict = defaultdict(list)
        for i, row in cache_df.iterrows():
            used_benchmarks_dict["Data set"].append(row["Data set"])
            used_benchmarks_dict["Preprocessing"].append(row["Preprocessing"])
            used_benchmarks_dict["Algorithm"].append(row["Algorithm"])
            used_benchmarks_dict["# Concepts"].append(row["# Concepts"])
            used_benchmarks_dict["# Words"].append(row["# Words"])
            used_benchmarks_dict["CUI Coverage"].append(row["CUI Coverage"])
            used_benchmarks_dict["UMLS Coverage"].append(row["UMLS Coverage"])
            used_benchmarks_dict[row["Benchmark"]].append(row["Score"])

        number_benchmarks = len(set(cache_df["Benchmark"]))
        reformat = ["Data set", "Algorithm", "Preprocessing", "# Concepts", "# Words", "CUI Coverage", "UMLS Coverage"]
        for column in reformat:
            used_benchmarks_dict[column] = [entry for i, entry in enumerate(used_benchmarks_dict[column])
                                            if i % number_benchmarks == 0]

        df_table = pd.DataFrame.from_dict(used_benchmarks_dict)
        df_table.to_csv(out_path, index=False, encoding="utf-8")
        return df_table

    def evaluate(self):
        tuples = []
        for embedding in self.embeddings:
            embedding.load()
            # vec_generator, dataset, algorithm, preprocessing = embedding
            # for vecs in vec_generator:
            cache_path = 'data/benchmark_cache.csv'
            for benchmark_class in self.benchmark_classes:
                benchmark = benchmark_class(embedding, self.umls_mapper, self.evaluators)
                score = benchmark.evaluate()

                german_cuis = set(benchmark.umls_mapper.umls_reverse_dict.keys())
                vocab_terms = set(benchmark.vocab.keys())
                actual_umls_terms = german_cuis.intersection(vocab_terms)
                nr_german_cuis = len(german_cuis)
                nr_vectors = len(vocab_terms)
                nr_concepts = len(actual_umls_terms)

                cui_cov = nr_concepts / nr_vectors  # ratio of found umls terms vs all vocab entries
                umls_cov = nr_concepts / nr_german_cuis  # ratio of found umls terms vs total UMLS terms
                # umls_cov = nr_concepts / len(benchmark.umls_mapper.umls_dict.keys())

                observation = (benchmark.dataset, benchmark.algorithm, benchmark.preprocessing, score,
                               nr_concepts, nr_vectors, cui_cov, umls_cov, benchmark.__class__.__name__,)
                tuples.append(observation)
                df_obs = pd.DataFrame([observation], columns=['Data set', 'Algorithm', 'Preprocessing', 'Score',
                                                              '# Concepts', '# Words', 'CUI Coverage',
                                                              'UMLS Coverage', 'Benchmark'])
                df_obs.to_csv(cache_path, mode='a', header=(not os.path.exists(cache_path)), index=False)
                benchmark.clean()
                del benchmark
            embedding.clean()
            del embedding

        df = pd.DataFrame(tuples, columns=['Data set', 'Algorithm', 'Preprocessing', 'Score', '# Concepts',
                                           '# Words', 'CUI Coverage', 'UMLS Coverage', 'Benchmark'])

        df.to_csv('data/benchmark_results1.csv', index=False, encoding="utf-8")
        df_table = Evaluation.build_paper_table(df, 'data/benchmark_results2.csv')
        print(df_table)


