import os

from benchmarking.benchmarks import *
from utils.transform_data import ConfigLoader
from vectorization.embeddings import Embeddings
from benchmarking.evaluation import Evaluation
from resource.other_resources import NDFEvaluator, SRSEvaluator
from collections import namedtuple


def main():
    config = ConfigLoader.get_config()

    umls_mapper = UMLSMapper(from_dir=config["path"]["UMLS"])
    Embedding = namedtuple('Embedding', 'vectors dataset algorithm preprocessing')

    # multi-term: sensible for multi token concepts
    # single-term: unsensible for multi token concepts
    # JULIE: JULIE repelacement of concepts
    # SE CUI: Subsequent Estimated CUIs with own method
    embeddings_to_benchmark = [

        # Related Work
        Embedding(Embeddings.load_w2v_format(os.path.join(config['PATH']['ExternalEmbeddings'], 'claims_cuis_hs_300.txt')), "Claims", "word2vec", "UNK"),
        Embedding(Embeddings.load_w2v_format(os.path.join(config['PATH']['ExternalEmbeddings'], 'DeVine_etal_200.txt')), "DeVine et al.", "word2vec", "UNK"),
        Embedding(Embeddings.load_w2v_format(os.path.join(config['PATH']['ExternalEmbeddings'], 'stanford_umls_svd_300.txt')),
                  "Stanford", "word2vec", "UNK"),
        Embedding(Embeddings.load_w2v_format(os.path.join(config['PATH']['ExternalEmbeddings'], 'cui2vec_pretrained.txt')), "cui2vec", "word2vec", "UNK"),
        Embedding(Embeddings.load(path="data/German_Medical.kv"), "GerVec", "word2vec", "multi-term"),

        # # Flair
        Embedding(Embeddings.assign_concepts_to_vecs(Embeddings.load(path="data/GGPONC_flair_no_cui_all.kv"),
                                                     umls_mapper),
                  "GGPONC", "Flair", "SE CUI"),
        Embedding(Embeddings.assign_concepts_to_vecs(Embeddings.load(path="data/German_Medical_flair_no_cui_all.kv"),
                                                     umls_mapper),
                  "GerVec", "Flair", "SE CUI"),
        Embedding(Embeddings.load(path="data/German_Medical_flair_JULIE_all.kv"),
                  "GerVec", "Flair", "JULIE"),
        Embedding(Embeddings.assign_concepts_to_vecs(
            Embeddings.load(path="data/German_Medical_bert_no_finetune_no_cui_all.kv"),
                                                     umls_mapper),
                  "GerVec", "BERT", "SE CUI NF"),
        Embedding(Embeddings.assign_concepts_to_vecs(
            Embeddings.load(path="data/60K_news_flair_no_cui_all.kv"),
            umls_mapper),
                  "News 100K", "Flair", "SE CUI"),
        Embedding(Embeddings.assign_concepts_to_vecs(
            Embeddings.load(path="data/60K_news_bert_no_finetune_no_cui_all.kv"),
            umls_mapper),
                  "News 100K", "BERT", "SE CUI NF"),
        Embedding(
            Embeddings.assign_concepts_to_vecs(
                Embeddings.load(path="data/German_Medical_flair_no_finetune_no_cui_all.kv"),
                umls_mapper),
            "GerVec", "Flair", "SE CUI NF"),
        Embedding(
            Embeddings.assign_concepts_to_vecs(
                Embeddings.load(path="data/100K_news_flair_no_fine_tune_no_cui_all.kv"),
                umls_mapper),
            "News 100K", "Flair", "SE CUI NF"),
        Embedding(
            Embeddings.assign_concepts_to_vecs(
                Embeddings.load(path="data/100K_news_flair_JULIE_all.kv"),
                umls_mapper),
            "News 100K", "Flair", "JULIE"),

        Embedding(Embeddings.load(path="data/100K_news_flair_all.kv"), "News 100K", "Flair", "multi-term"),

        # Gervec / news multiterm flair, single term flair

        # # GGPONC
        Embedding(Embeddings.load(path="data/no_prep_vecs_test_all.kv"), "GGPONC", "word2vec", "multi-term"),
        # Embedding(Embeddings.load(path="data/GGPONC_plain_all.kv"), "GGPONC", "word2vec", "single-term"),
        # Embedding(Embeddings.load(path="data/GGPONC_JULIE_all.kv"), "GGPONC", "word2vec", "JULIE"),
        Embedding(Embeddings.assign_concepts_to_vecs(Embeddings.load(path="data/GGPONC_no_cui_all.kv"), umls_mapper),
                  "GGPONC", "word2vec", "SE CUI"),
        # Embedding(Embeddings.load(path="data/GGPONC_fastText_all.kv"), "GGPONC", "fastText", "multi-term"),
        # Embedding(Embeddings.load(path="data/GGPONC_glove_all.kv"), "GGPONC", "Glove", "multi-term"),
        # # Pretrained
        # # https://devmount.github.io/GermanWordEmbeddings/
        # Embedding(Embeddings.assign_concepts_to_vecs(Embeddings.load_w2v_format('E:/german.model', binary=True),
        #                                   umls_mapper), "Wikipedia + News 2015", "word2vec", "SE CUI"),
        #
        # # News
        # Embedding(Embeddings.load(path="data/60K_news_all.kv"), "News 60K", "word2vec", "multi-term"),
        # Embedding(Embeddings.load(path="data/60K_news_plain_all.kv"), "News 60K", "word2vec", "single-term"),
        # Embedding(Embeddings.load(path="data/60K_news_JULIE_all.kv"), "News 60K", "word2vec", "JULIE"),
        # Embedding(Embeddings.assign_concepts_to_vecs(Embeddings.load(path="data/60K_news_no_cui_all.kv"), umls_mapper),
        #           "News 60K", "word2vec", "SE CUI"),
        # Embedding(Embeddings.load(path="data/60K_news_all.kv"), "News 60K", "fastText", "multi-term"),
        # Embedding(Embeddings.load(path="data/60K_news_glove_all.kv"), "News 60K", "Glove", "multi-term"),
        # Embedding(Embeddings.load(path="data/500K_news_all.kv"), "News 500K", "word2vec", "multi-term"),
        # # Embedding(Embeddings.load(path="data/3M_news_all.kv"), "News 3M", "word2vec", "multi-term"),
        #
        # # JSynCC
        Embedding(Embeddings.load(path="data/JSynCC_all.kv"), "JSynCC", "word2vec", "multi-term"),
        #
        # # PubMed
        Embedding(Embeddings.load(path="data/PubMed_all.kv"), "PubMed", "word2vec", "multi-term"),
        #
        # # German Medical Concatenation
        # Embedding(Embeddings.load(path="data/German_Medical_all.kv"), "GerVec", "word2vec", "multi-term"),
        # Embedding(Embeddings.load(path="data/German_Medical_plain_all.kv"), "GerVec", "word2vec", "single-term"),
        # Embedding(Embeddings.load(path="data/German_Medical_JULIE_all.kv"), "GerVec", "word2vec", "JULIE"),
        # Embedding(Embeddings.assign_concepts_to_vecs(
        #     Embeddings.load(path="data/German_Medical_no_cui_all.kv"), umls_mapper),
        #     "GerVec", "word2vec", "SE CUI"),
        # Embedding(Embeddings.load(path="data/German_Medical_fastText_all.kv"), "GerVec", "fastText", "multi-term"),
        # Embedding(Embeddings.load(path="data/German_Medical_fastText_plain_all.kv"), "GerVec",
        #           "fastText", "single-term"),
        # Embedding(Embeddings.load(path="data/German_Medical_fastText_JULIE_all.kv"), "GerVec",
        #           "fastText", "JULIE"),
        # Embedding(Embeddings.assign_concepts_to_vecs(Embeddings.load(path="data/German_Medical_fastText_no_cui_all.kv"),
        #                                   umls_mapper), "GerVec", "fastText", "SE CUI"),
        # Embedding(Embeddings.load(path="data/German_Medical_all.kv"), "GerVec", "Glove", "multi-term"),
        # Embedding(Embeddings.load(path="data/German_Medical_Glove_plain_all.kv"), "GerVec",
        #           "Glove", "single-term"),
        # Embedding(Embeddings.load(path="data/German_Medical_Glove_JULIE_all.kv"), "GerVec",
        #           "Glove", "JULIE"),
        # Embedding(Embeddings.assign_concepts_to_vecs(
        #     Embeddings.load(path="data/German_Medical_Glove_no_cui_all.kv"), umls_mapper),
        #     "GerVec", "Glove", "SE CUI")
    ]



    evaluators = [
        UMLSEvaluator(from_dir=config["PATH"]["UMLS"]),
        NDFEvaluator(from_dir=config["PATH"]["NDF"]),
        SRSEvaluator(from_dir=config["PATH"]["SRS"]),
        MRRELEvaluator(from_dir=config["PATH"]["UMLS"])
    ]

    benchmarks_to_use = [
        HumanAssessment,
        # CategoryBenchmark,
        # SilhouetteCoefficient,
        CausalityBeam,
        NDFRTBeam,
        SemanticTypeBeam,
        AssociationBeam,
        ConceptualSimilarityChoi,
        # MedicalRelatednessMayTreatChoi,
        # MedicalRelatednessMayPreventChoi
    ]

    Evaluation(embeddings_to_benchmark,
               umls_mapper,
               evaluators,
               benchmarks_to_use)

    # benchmark.category_benchmark("Nucleotide Sequence")
    # emb = EmbeddingSet({umls_mapper.un_umls(c, single_return=True):
    #                         Embedding(umls_mapper.un_umls(c, single_return=True), ggponc_vecs[c])
    #                     for c in ggponc_vecs.vocab})
    # emb = EmbeddingSet({c: Embedding(c, ggponc_vecs[c]) for c in ggponc_vecs.vocab})
    # emb.plot_interactive("Fibroblasten", "Fremdkörper")


if __name__ == "__main__":
    main()
