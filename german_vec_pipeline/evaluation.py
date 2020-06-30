from collections import defaultdict
from typing import Tuple, List

import gensim

from german_vec_pipeline.UMLS import UMLSMapper
from german_vec_pipeline.benchmarks import Benchmark
from german_vec_pipeline.resources import Evaluator

import pandas as pd


class Evaluation:
    def __init__(self, embeddings: List[Tuple[gensim.models.KeyedVectors, str, str, str]],
                 umls_mapper: UMLSMapper,
                 evaluators: List[Evaluator],
                 benchmark_classes=List[Benchmark]):

        self.benchmarks = [benchmark(embedding, umls_mapper, evaluators)
                           for embedding in embeddings
                           for benchmark in benchmark_classes]

        self.evaluate()

    def evaluate(self):
        tuples = []
        for benchmark in self.benchmarks:
            # print(benchmark.__class__.__name__, benchmark.dataset, benchmark.algorithm)
            score = benchmark.evaluate()
            german_cuis = set(benchmark.umls_mapper.umls_reverse_dict.keys())
            vocab_terms = set(benchmark.vocab.keys())
            actual_umls_terms = german_cuis.intersection(vocab_terms)
            nr_german_cuis = len(german_cuis)
            nr_vectors = len(vocab_terms)
            nr_concepts = len(actual_umls_terms)

            cui_cov = nr_concepts / nr_vectors   # ratio of found umls terms vs all vocab entries
            umls_cov = nr_concepts / nr_german_cuis  # ratio of found umls terms vs total UMLS terms
            # umls_cov = nr_concepts / len(benchmark.umls_mapper.umls_dict.keys())

            tuples.append((benchmark.dataset, benchmark.algorithm, benchmark.preprocessing, score,
                           nr_concepts, nr_vectors, cui_cov, umls_cov, benchmark.__class__.__name__, ))

        df = pd.DataFrame(tuples, columns=['Data set', 'Algorithm', 'Preprocessing', 'Score', '# Concepts',
                                           '# Words', 'CUI Coverage', 'UMLS Coverage', 'Benchmark'])
        # df["CUI Coverage"] = (df["# Concepts"] / df["# Words"])
        print(df)
        df.to_csv('data/benchmark_results1.csv', index=False, encoding="utf-8")
        used_benchmarks_dict = defaultdict(list)
        for i, row in df.iterrows():
            used_benchmarks_dict["Data set"].append(row["Data set"])
            used_benchmarks_dict["Preprocessing"].append(row["Preprocessing"])
            used_benchmarks_dict["Algorithm"].append(row["Algorithm"])
            used_benchmarks_dict["# Concepts"].append(row["# Concepts"])
            used_benchmarks_dict["# Words"].append(row["# Words"])
            used_benchmarks_dict["CUI Coverage"].append(row["CUI Coverage"])
            used_benchmarks_dict["UMLS Coverage"].append(row["UMLS Coverage"])
            used_benchmarks_dict[row["Benchmark"]].append(row["Score"])

        number_benchmarks = len(set(df["Benchmark"]))
        reformat = ["Data set", "Algorithm", "Preprocessing", "# Concepts", "# Words", "CUI Coverage", "UMLS Coverage"]
        for column in reformat:
            used_benchmarks_dict[column] = [entry for i, entry in enumerate(used_benchmarks_dict[column])
                                            if i % number_benchmarks == 0]

        df_table = pd.DataFrame.from_dict(used_benchmarks_dict)

        print(df_table)
        df_table.to_csv('data/benchmark_results2.csv', index=False, encoding="utf-8")
