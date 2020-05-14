import gensim
from whatlies import EmbeddingSet, Embedding

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

    def category_benchmark(self):
        choosen_category = "Nucleotide Sequence"
        other_categories = self.umls_evaluator.category2concepts.keys()
        p1 = self.pairwise_cosine(self.umls_evaluator.category2concepts[choosen_category])
        print(p1)

        p2s = []
        for other_category in other_categories:
            if other_category == choosen_category:
                continue
            choosen_concepts = self.umls_evaluator.category2concepts[choosen_category]
            other_concepts = self.umls_evaluator.category2concepts[other_category]
            if len(choosen_concepts) == 0 or len(other_concepts) == 0:
                continue
            p2 = self.pairwise_cosine(choosen_concepts, other_concepts)
            p2s.append(p2)

        avg_p2 = sum(p2s) / len(p2s)
        print(p2s)
        print(p1, avg_p2, p1 - avg_p2)
        return p1 - avg_p2


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
    vecs = Embeddings.load(path="E:/AML4DHGermanVecs/test_vecs_1.kv")

    for c, v in analogies(vecs, "Asthma", "Lunge", "Herz", umls=umls_mapper):
        print(umls_mapper.un_umls(c, single_return=True), v)

    for c, v in similarities(vecs, "Hepatitis", umls=umls_mapper):
        print(umls_mapper.un_umls(c, single_return=True), v)

    for c, v in similarities(vecs, "Cisplatin", umls=umls_mapper):
        print(umls_mapper.un_umls(c, single_return=True), v)

    # print([(umls_mapper.un_umls(c), Embedding(umls_mapper.un_umls(c), vecs[c])) for c in vecs.vocab])
    evaluator = UMLSEvaluator(vectors=vecs)
    benchmark = Benchmark(vecs, umls_mapper, evaluator)

    benchmark.category_benchmark()

    p1 = benchmark.pairwise_cosine(evaluator.category2concepts["Medical Device"])
    print(p1)
    p2 = benchmark.pairwise_cosine(evaluator.category2concepts["Medical Device"],
                                   evaluator.category2concepts["Health Care Related Organization"])
    print(p2)

    emb = EmbeddingSet(
        {umls_mapper.un_umls(c, single_return=True): Embedding(umls_mapper.un_umls(c, single_return=True), vecs[c]) for
         c in vecs.vocab})
    # emb = EmbeddingSet({c: Embedding(c, vecs[c]) for c in vecs.vocab})

    emb.plot_interactive("Fibroblasten", "Fremdkörper")

    # replace multi words


if __name__ == "__main__":
    main()
