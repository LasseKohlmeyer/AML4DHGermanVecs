import math

import gensim
from tqdm import tqdm

from UMLS import UMLSMapper, UMLSEvaluator
from embeddings import Embeddings


class Benchmark:
    def __init__(self, embeddings: gensim.models.KeyedVectors, umls_mapper: UMLSMapper, umls_evaluator: UMLSEvaluator):
        self.embeddings = embeddings
        self.umls_mapper = umls_mapper
        self.umls_evaluator = umls_evaluator

    def pairwise_cosine(self, concepts1, concepts2=None):
        def cosine(word1=None, word2=None, c1=None, c2=None):
            if word1:
                return self.embeddings.similarity(self.umls_mapper.umls_dict[word1], self.umls_mapper.umls_dict[word2])
            else:
                return self.embeddings.similarity(c1, c2)

        if concepts2 is None:
            concepts2 = concepts1
            s = 0
            count = 0
            for i, concept1 in enumerate(concepts1):
                for j, concept2 in enumerate(concepts2):
                    if j > i:
                        c = cosine(c1=concept1, c2=concept2)
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
                    c = cosine(c1=concept1, c2=concept2)
                    if c < 0:
                        c = -c
                    s += c
                    count += 1
            return s / count

    def category_benchmark(self, choosen_category="Nucleotide Sequence"):
        other_categories = self.umls_evaluator.category2concepts.keys()
        choosen_concepts = self.umls_evaluator.category2concepts[choosen_category]
        if len(choosen_concepts) <= 1:
            return 0
        p1 = self.pairwise_cosine(choosen_concepts)

        p2s = []
        for other_category in other_categories:
            if other_category == choosen_category:
                continue

            other_concepts = self.umls_evaluator.category2concepts[other_category]
            if len(choosen_concepts) == 0 or len(other_concepts) == 0:
                continue
            p2 = self.pairwise_cosine(choosen_concepts, other_concepts)
            p2s.append(p2)

        avg_p2 = sum(p2s) / len(p2s)
        # print(p2s)
        print(f'{choosen_category}: within {p1}, out {avg_p2}, distance {p1 - avg_p2}')
        return p1 - avg_p2

    def all_categories_benchmark(self):

        distances = []
        cat = ""
        for category in tqdm(self.umls_evaluator.category2concepts.keys()):
            distance_within_without = self.category_benchmark(category)
            distances.append(distance_within_without)

        benchmark_value = sum(distances) / len(distances)
        print(benchmark_value)
        return benchmark_value

    def mcsm_umls(self, category, k=40):
        def category_true(concept, category):
            if category in self.umls_evaluator.concept2category[concept]:
                return 1
            else:
                return 0

        v_t = self.umls_evaluator.category2concepts[category]
        if len(v_t) == 0:
            return 0

        sigma = 0
        for v in v_t:
            for i in range(0, k):
                neighbors = self.embeddings.most_similar(v, topn=k)
                v_i = neighbors[i][0]
                sigma += category_true(v_i, category) / math.log((i + 1) + 1, 2)
        return sigma / len(v_t)

    def choi_benchmark(self):
        print(self.mcsm_umls("Pharmacologic Substance"))
        print(self.mcsm_umls("Disease or Syndrome"))
        print(self.mcsm_umls("Neoplastic Process"))
        print(self.mcsm_umls("Clinical Drug"))
        print(self.mcsm_umls("Finding"))
        print(self.mcsm_umls("Injury or Poisoning"))


def analogies(vectors, start, minus, plus, umls: UMLSMapper):
    if umls:
        return vectors.most_similar(positive=[umls.umls_dict[start], umls.umls_dict[plus]],
                                    negative=[umls.umls_dict[minus]])
    else:
        return vectors.most_similar(positive=[start, plus], negative=[minus])


def similarities(vectors, word, umls):
    if umls:
        return vectors.most_similar(umls.umls_dict[word])
    else:
        return vectors.most_similar(word)


def main():
    umls_mapper = UMLSMapper(from_dir='E:/AML4DH-DATA/UMLS')
    vecs = Embeddings.load(path="data/no_prep_vecs.kv")
    evaluator = UMLSEvaluator(from_dir='E:/AML4DH-DATA/UMLS', vectors=vecs)

    # for c, v in vecs.most_similar("Cisplatin"):
    #     print(umls_mapper.un_umls(c), v)
    #
    #
    # # for c, v in vecs.most_similar(umls_mapper.umls_dict["Cisplatin"]):
    # #     print(umls_mapper.un_umls(c), v)
    #
    # for c, v in analogies(vecs, "Asthma", "Lunge", "Herz", umls=umls_mapper):
    #     print(umls_mapper.un_umls(c), v)
    #
    # for c, v in similarities(vecs, "Hepatitis", umls=umls_mapper):
    #     print(umls_mapper.un_umls(c), v)
    #
    # for c, v in similarities(vecs, "Cisplatin", umls=umls_mapper):
    #     print(umls_mapper.un_umls(c), v)

    # print([(umls_mapper.un_umls(c), Embedding(umls_mapper.un_umls(c), vecs[c])) for c in vecs.vocab])

    benchmark = Benchmark(vecs, umls_mapper, evaluator)
    benchmark.choi_benchmark()
    # benchmark.all_categories_benchmark()
    # benchmark.category_benchmark()

    # emb = EmbeddingSet( {umls_mapper.un_umls(c, single_return=True): Embedding(umls_mapper.un_umls(c,
    # single_return=True), vecs[c]) for c in vecs.vocab}) # emb = EmbeddingSet({c: Embedding(c, vecs[c]) for c in
    # vecs.vocab})
    #
    # emb.plot_interactive("Fibroblasten", "Fremdkörper")

    # replace multi words


if __name__ == "__main__":
    main()
