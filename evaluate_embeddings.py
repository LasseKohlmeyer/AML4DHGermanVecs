# from gensim.models.wrappers import FastText
from gensim.models.fasttext import load_facebook_model

from benchmarks import *
from embeddings import Embeddings
from evaluation_resource import NDFEvaluator, SRSEvaluator
import pandas as pd


class Evaluation:
    def __init__(self, embeddings: List[Tuple[gensim.models.KeyedVectors, str, str]],
                 umls_mapper: UMLSMapper,
                 umls_evaluator: UMLSEvaluator,
                 ndf_evaluator: NDFEvaluator,
                 srs_evaluator: SRSEvaluator,
                 benchmark_classes=List[Benchmark]):

        self.benchmarks = []
        for embedding in embeddings:
            if CategoryBenchmark in benchmark_classes:
                self.benchmarks.append(CategoryBenchmark(embedding, umls_mapper, umls_evaluator))
            if SemanticTypeBeam in benchmark_classes:
                self.benchmarks.append(SemanticTypeBeam(embedding, umls_mapper, umls_evaluator))
            if SilhouetteCoefficient in benchmark_classes:
                self.benchmarks.append(SilhouetteCoefficient(embedding, umls_mapper, umls_evaluator))
            if ChoiBenchmark in benchmark_classes:
                self.benchmarks.append(ChoiBenchmark(embedding, umls_mapper, umls_evaluator, ndf_evaluator))
            if HumanAssessment in benchmark_classes:
                self.benchmarks.append(HumanAssessment(embedding, umls_mapper, srs_evaluator))

    def evaluate(self):
        tuples = []
        for benchmark in self.benchmarks:
            print(benchmark.__class__.__name__, benchmark.dataset, benchmark.algorithm)
            score = benchmark.evaluate()
            nr_concepts = len(set(benchmark.umls_mapper.umls_reverse_dict.keys()).intersection(set(benchmark.vocab)))
            nr_vectors = len(benchmark.vocab)
            tuples.append((benchmark.dataset, benchmark.algorithm, benchmark.__class__.__name__, score,
                           nr_concepts, nr_vectors))

        df = pd.DataFrame(tuples, columns=['Data set', 'Algorithm', 'Benchmark', 'Score', '# Concepts', '# Words'])
        df["CUI Coverage"] = (df["# Concepts"] / df["# Words"])
        print(df)
        df.to_csv('data/benchmark_results1.csv', index=False, encoding="utf-8")
        used_benchmarks_dict = defaultdict(list)
        for i, row in df.iterrows():
            used_benchmarks_dict["Data set"].append(row["Data set"])
            used_benchmarks_dict["Algorithm"].append(row["Algorithm"])
            used_benchmarks_dict["# Concepts"].append(row["# Concepts"])
            used_benchmarks_dict["# Words"].append(row["# Words"])
            used_benchmarks_dict[row["Benchmark"]].append(row["Score"])
            used_benchmarks_dict["CUI Coverage"].append(row["CUI Coverage"])

        number_benchmarks = len(set(df["Benchmark"]))
        reformat = ["Data set", "Algorithm", "# Concepts", "# Words", "CUI Coverage"]
        for column in reformat:
            used_benchmarks_dict[column] = [entry for i, entry in enumerate(used_benchmarks_dict[column])
                                            if i % number_benchmarks == 0]

        df_table = pd.DataFrame.from_dict(used_benchmarks_dict)

        print(df_table)
        df_table.to_csv('data/benchmark_results2.csv', index=False, encoding="utf-8")

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
    ggponc_vecs_fasttext = (Embeddings.load(path="data/GGPONC_fastText_all.kv"), "GGPONC", "fastText")
    ggponc_vecs_glove = (Embeddings.load(path="data/GGPONC_glove_all.kv"), "GGPONC", "Glove")
    ggponc_vecs = (Embeddings.load(path="data/no_prep_vecs_test_all.kv"), "GGPONC", "word2vec")
    ggponc_vecs_julie = (Embeddings.load(path="data/GGPONC_JULIE_all.kv"), "GGPONC JULIE", "word2vec")
    ggponc_vecs_no_cui = (assign_concepts_to_vecs(Embeddings.load(path="data/GGPONC_no_cui_all.kv"), umls_mapper),
                          "GGPONC NO CUI", "word2vec")
    ggponc_vecs_plain = (Embeddings.load(path="data/GGPONC_plain_all.kv"), "GGPONC plain", "word2vec")

    # https://devmount.github.io/GermanWordEmbeddings/
    # pretrained_wiki_news_vecs = (word2vec.KeyedVectors.load_word2vec_format('E:/german.model', binary=True),
    #                              "Wikipedia + News 2015", "word2vec")
    #
    # pretrained_wiki_news_vecs = (assign_concepts_to_vecs(pretrained_wiki_news_vecs[0], umls_mapper),
    #                              "Wikipedia + News 2015", "word2vec")

    news_vecs = (Embeddings.load(path="data/60K_news_all.kv"), "News 60K", "word2vec")
    news_vecs_big = (Embeddings.load(path="data/500K_news_all.kv"), "News 500K", "word2vec")
    # news_vecs_big_3M = (Embeddings.load(path="data/3M_news_all.kv"), "News 3M", "word2vec")
    news_vecs_fasttext = (Embeddings.load(path="data/60K_news_all.kv"), "News 60K", "fastText")
    news_vecs_glove = (Embeddings.load(path="data/60K_news_glove_all.kv"), "News 60K", "Glove")
    news_vecs_julie = (Embeddings.load(path="data/60K_news_JULIE_all.kv"), "News 60K JULIE", "word2vec")
    news_vecs_no_cui = (assign_concepts_to_vecs(Embeddings.load(path="data/60K_news_no_cui_all.kv"), umls_mapper),
                        "News 60K NO CUI", "word2vec")
    news_vecs_plain = (Embeddings.load(path="data/60K_news_plain_all.kv"), "News 60K plain", "word2vec")

    jsyncc_vecs = (Embeddings.load(path="data/JSynCC_all.kv"), "JSynCC", "word2vec")
    pubmed_vecs = (Embeddings.load(path="data/PubMed_all.kv"), "PubMed", "word2vec")

    # fasttext_model = load_fasttext_model('E://cc.de.300.bin')
    # fasttext_vecs = (load_facebook_model('E:/cc.de.300.bin'), "common crawl", "fastText")

    umls_evaluator = UMLSEvaluator(from_dir='E:/AML4DH-DATA/UMLS')

    ndf_evaluator = NDFEvaluator(from_dir='E:/AML4DH-DATA/NDF')

    srs_evaluator = SRSEvaluator(from_dir="E:/AML4DH-DATA/SRS")

    benchmarks_to_use = [
        HumanAssessment,
        CategoryBenchmark,
        # SemanticTypeBeam,
        # SilhouetteCoefficient,
        # ChoiBenchmark
    ]

    evaluation = Evaluation([
                             news_vecs, news_vecs_big, news_vecs_fasttext, news_vecs_glove,
                             news_vecs_julie, news_vecs_no_cui, news_vecs_plain,
                             ggponc_vecs, ggponc_vecs_fasttext,
                             ggponc_vecs_glove,
                             ggponc_vecs_julie, ggponc_vecs_no_cui, ggponc_vecs_plain,
                             jsyncc_vecs, pubmed_vecs
                            ],
                            umls_mapper, umls_evaluator, ndf_evaluator, srs_evaluator, benchmarks_to_use)
    evaluation.evaluate()

    # benchmark.category_benchmark("Nucleotide Sequence")
    # emb = EmbeddingSet({umls_mapper.un_umls(c, single_return=True):
    #                         Embedding(umls_mapper.un_umls(c, single_return=True), ggponc_vecs[c])
    #                     for c in ggponc_vecs.vocab})
    # emb = EmbeddingSet({c: Embedding(c, ggponc_vecs[c]) for c in ggponc_vecs.vocab})
    # emb.plot_interactive("Fibroblasten", "Fremdkörper")


if __name__ == "__main__":
    main()
