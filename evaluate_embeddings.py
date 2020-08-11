import os
from benchmarking.benchmarks import *
from utils.transform_data import ConfigLoader
from vectorization.embeddings import Embeddings, Embedding
from benchmarking.evaluation import Evaluation
from resource.other_resources import NDFEvaluator, SRSEvaluator
import pandas as pd

def main():
    config = ConfigLoader.get_config()
    umls_mapper = Embeddings.set_config_and_get_umls_mapper(config)
    # umls_mapper = UMLSMapper(from_dir=config["PATH"]["UMLS"])
    # Embeddings.set_umls_mapper(umls_mapper)
    # Embedding = namedtuple('Embedding', 'vectors dataset algorithm preprocessing')

    # multi-term: sensible for multi token concepts
    # single-term: unsensible for multi token concepts
    # JULIE: JULIE repelacement of concepts
    # SE CUI: Subsequent Estimated CUIs with own method
    # e = Embedding('German_Medical_Glove_JULIE_NEW_all.kv', "GerVec", "GloVe", "JCoRe")
    # e.load()
    # print(e.vectors["C0035647"])
    embeddings_to_benchmark = [
        # # # Related Work
        Embedding('German_Medical.kv', "GerVec", "word2vec", "multi-term"),
        Embedding('stanford_umls_svd_300.txt', "Choi et al. 2016 [Stanford]", "word2vec", "UNK", internal=False),
        Embedding('claims_cuis_hs_300.txt', "Choi et al. 2016 [Claims]", "word2vec", "UNK", internal=False),
        Embedding('DeVine_etal_200.txt', "De Vine et al. 2014", "word2vec", "UNK", internal=False),
        Embedding('cui2vec_pretrained.txt', "Beam et al. 2020", "word2vec", "UNK", internal=False),
        #
        # # # Flair
        Embedding('GGPONC_flair_no_cui_all.kv', "GGPONC", "Flair", "SE CUI", estimate_cui=True),

        Embedding('German_Medical_flair_all.kv', "GerVec", "Flair", "multi-term"),
        Embedding('German_Medical_flair_plain_all.kv', "GerVec", "Flair", "single-term"),
        Embedding('German_Medical_flair_JULIE_all.kv', "GerVec", "Flair", "JCoRe"),
        Embedding('German_Medical_flair_no_cui_all.kv', "GerVec", "Flair", "SE CUI", estimate_cui=True),
        Embedding('German_Medical_flair_no_finetune_no_cui_all.kv', "GerVec", "Flair", "SE CUI NF", estimate_cui=True),
        Embedding('German_Medical_bert_no_finetune_no_cui_all.kv', "GerVec", "BERT", "SE CUI NF", estimate_cui=True),

        Embedding('100K_news_flair_all.kv', "News 100K", "Flair", "multi-term"),
        Embedding('100K_news_flair_plain_all.kv', "News 100K", "Flair", "single-term"),
        Embedding('100K_news_flair_JULIE_all.kv', "News 100K", "Flair", "JCoRe"),
        Embedding('100K_news_flair_no_cui_all.kv', "News 100K", "Flair", "SE CUI", estimate_cui=True),
        Embedding('100K_news_flair_no_fine_tune_no_cui_all.kv', "News 100K", "Flair", "SE CUI NF", estimate_cui=True),
        Embedding('100K_news_bert_no_finetune_no_cui_all.kv', "News 100K", "BERT", "SE CUI NF", estimate_cui=True),
        #
        # # Gervec / news multiterm flair, single term flair
        # # # GGPONC
        Embedding('GGPONC_all.kv', "GGPONC", "word2vec", "multi-term"),
        Embedding('GGPONC_fastText_all.kv', "GGPONC", "fastText", "multi-term"),
        Embedding('GGPONC_glove_all.kv', "GGPONC", "GloVe", "multi-term"),
        Embedding('GGPONC_plain_all.kv', "GGPONC", "word2vec", "single-term"),
        Embedding('GGPONC_fastText_plain_all.kv',"GGPONC", "fastText", "single-term"),
        Embedding('GGPONC_glove_plain_all.kv', "GGPONC", "GloVe", "single-term"),
        Embedding('GGPONC_no_cui_all.kv', "GGPONC", "word2vec", "SE CUI", estimate_cui=True),
        Embedding('GGPONC_fastText_no_cui_all.kv', "GGPONC", "fastText", "SE CUI", estimate_cui=True),
        Embedding('GGPONC_glove_no_cui_all.kv', "GGPONC", "GloVe", "SE CUI", estimate_cui=True),
        Embedding('GGPONC_JULIE_all.kv', "GGPONC", "word2vec", "JCoRe"),
        Embedding('GGPONC_fastText_JULIE_all.kv', "GGPONC", "fastText", "JCoRe"),
        Embedding('GGPONC_glove_JULIE_all.kv', "GGPONC", "GloVe", "JCoRe"),
        # # Pretrained
        # # https://devmount.github.io/GermanWordEmbeddings/
        Embedding('E:/german.model', "Wikipedia + News 2015", "word2vec", "SE CUI",
                  internal=False, estimate_cui=True, is_file=False),
        # # News
        Embedding('100K_news_all.kv', "News 100K", "word2vec", "multi-term"),
        Embedding('100K_news_fastText_all.kv', "News 100K", "fastText", "multi-term"),
        Embedding('100K_news_glove_all.kv', "News 100K", "GloVe", "multi-term"),
        Embedding('100K_news_plain_all.kv', "News 100K", "word2vec", "single-term"),
        Embedding('100K_news_fastText_plain_all.kv', "News 100K", "fastText", "single-term"),
        Embedding('100K_news_glove_plain_all.kv', "News 100K", "GloVe", "single-term"),
        Embedding('100K_news_no_cui_all.kv', "News 100K", "word2vec", "SE CUI", estimate_cui=True),
        Embedding('100K_news_fastText_no_cui_all.kv', "News 100K", "fastText", "SE CUI", estimate_cui=True),
        Embedding('100K_news_glove_no_cui_all.kv', "News 100K", "GloVe", "SE CUI", estimate_cui=True),
        Embedding('100K_news_JULIE_all.kv', "News 100K", "word2vec", "JCoRe"),
        Embedding('100K_news_fastText_JULIE_all.kv', "News 100K", "fastText", "JCoRe"),
        Embedding('100K_news_glove_JULIE_all.kv', "News 100K", "GloVe", "JCoRe"),

        Embedding('500K_news_all.kv', "News 500K", "word2vec", "multi-term"),
        Embedding('3M_news_all.kv', "News 3M", "word2vec", "multi-term"),

        # # JSynCC
        Embedding('JSynnCC_all.kv', "JSynCC", "word2vec", "multi-term"),
        Embedding('JSynnCC_fastText_all.kv', "JSynCC", "fastText", "multi-term"),
        Embedding('JSynnCC_glove_all.kv', "JSynCC", "GloVe", "multi-term"),
        Embedding('JSynnCC_plain_all.kv', "JSynCC", "word2vec", "single-term"),
        Embedding('JSynnCC_fastText_plain_all.kv', "JSynCC", "fastText", "single-term"),
        Embedding('JSynnCC_glove_plain_all.kv', "JSynCC", "GloVe", "single-term"),
        Embedding('JSynnCC_no_cui_all.kv', "JSynCC", "word2vec", "SE CUI", estimate_cui=True),
        Embedding('JSynnCC_fastText_no_cui_all.kv', "JSynCC", "fastText", "SE CUI", estimate_cui=True),
        Embedding('JSynnCC_glove_no_cui_all.kv', "JSynCC", "GloVe", "SE CUI", estimate_cui=True),
        Embedding('JSynnCC_JULIE_all.kv', "JSynCC", "word2vec", "JCoRe"),
        Embedding('JSynnCC_fastText_JULIE_all.kv', "JSynCC", "fastText", "JCoRe"),
        Embedding('JSynnCC_glove_JULIE_all.kv', "JSynCC", "GloVe", "JCoRe"),
        # # # PubMed
        Embedding('PubMed_all.kv', "PubMed", "word2vec", "multi-term"),
        Embedding('PubMed_fastText_all.kv', "PubMed", "fastText", "multi-term"),
        Embedding('PubMed_glove_all.kv', "PubMed", "GloVe", "multi-term"),
        Embedding('PubMed_plain_all.kv', "PubMed", "word2vec", "single-term"),
        Embedding('PubMed_fastText_plain_all.kv', "PubMed", "fastText", "single-term"),
        Embedding('PubMed_glove_plain_all.kv', "PubMed", "GloVe", "single-term"),
        Embedding('PubMed_no_cui_all.kv', "PubMed", "word2vec", "SE CUI", estimate_cui=True),
        Embedding('PubMed_fastText_no_cui_all.kv',  "PubMed", "fastText", "SE CUI", estimate_cui=True),
        Embedding('PubMed_glove_no_cui_all.kv', "PubMed", "GloVe", "SE CUI", estimate_cui=True),
        Embedding('PubMed_JULIE_all.kv', "PubMed", "word2vec", "JCoRe"),
        Embedding('PubMed_fastText_JULIE_all.kv', "PubMed", "fastText", "JCoRe"),
        Embedding('PubMed_glove_JULIE_all.kv', "PubMed", "GloVe", "JCoRe"),
        # # # German Medical Concatenation
        Embedding('German_Medical_all.kv', "GerVec", "word2vec", "multi-term"),
        Embedding('German_Medical_fastText_all.kv', "GerVec", "fastText", "multi-term"),
        Embedding('German_Medical_Glove_all.kv', "GerVec", "GloVe", "multi-term"),
        Embedding('German_Medical_plain_all.kv', "GerVec", "word2vec", "single-term"),
        Embedding('German_Medical_fastText_plain_all.kv', "GerVec", "fastText", "single-term"),
        Embedding('German_Medical_Glove_plain_all.kv', "GerVec", "GloVe", "single-term"),
        Embedding('German_Medical_no_cui_all.kv', "GerVec", "word2vec", "SE CUI", estimate_cui=True),
        Embedding('German_Medical_fastText_no_cui_all.kv', "GerVec", "fastText", "SE CUI", estimate_cui=True),
        Embedding('German_Medical_Glove_no_cui_all.kv', "GerVec", "GloVe", "SE CUI", estimate_cui=True),
        Embedding('German_Medical_JULIE_all.kv', "GerVec", "word2vec", "JCoRe"),
        Embedding('German_Medical_fastText_JULIE_all.kv', "GerVec", "fastText", "JCoRe"),
        Embedding('German_Medical_Glove_JULIE_all.kv', "GerVec", "GloVe", "JCoRe"),

        # Embedding('German_Medical_Glove_JULIE_P_all.kv', "GerVec", "GloVe", "JCoRe"),
        # Embedding('German_Medical_Glove_JULIE_P.kv', "GerVec REL", "GloVe", "JCoRe")
    ]

    # German_Medical_Glove_JULIE_al
    # new glove julie
    # GGONC single term, multi term

    evaluators = [
        UMLSEvaluator(from_dir=config["PATH"]["UMLS"]),
        NDFEvaluator(from_dir=config["PATH"]["NDF"]),
        SRSEvaluator(from_dir=config["PATH"]["SRS"]),
        MRRELEvaluator(from_dir=config["PATH"]["UMLS"])
    ]

    benchmarks_to_use = [
        # HumanAssessmentRelatedness,
        HumanAssessmentRelatednessCont,
        HumanAssessmentSimilarityCont,
        HumanAssessmentMayoSRS,
        # CausalityBeam,
        # NDFRTBeam,
        # SemanticTypeBeam,
        # AssociationBeam,
        # ConceptualSimilarityChoi,

        # CategoryBenchmark,
        # SilhouetteCoefficient,
        # MedicalRelatednessMayTreatChoi,
        # MedicalRelatednessMayPreventChoi
    ]

    Evaluation(embeddings_to_benchmark,
               umls_mapper,
               evaluators,
               benchmarks_to_use)

    Evaluation.build_paper_table(pd.read_csv('data/benchmark_cache.csv'), "data/benchmark_table_from_cache.csv")
    # benchmark.category_benchmark("Nucleotide Sequence")
    # emb = EmbeddingSet({umls_mapper.un_umls(c, single_return=True):
    #                         Embedding(umls_mapper.un_umls(c, single_return=True), ggponc_vecs[c])
    #                     for c in ggponc_vecs.vocab})
    # emb = EmbeddingSet({c: Embedding(c, ggponc_vecs[c]) for c in ggponc_vecs.vocab})
    # emb.plot_interactive("Fibroblasten", "Fremdkörper")


if __name__ == "__main__":
    main()
