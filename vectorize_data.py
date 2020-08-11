import os
from vectorization.embeddings import Embeddings
from utils.transform_data import DataHandler, ConfigLoader


def main():
    config = ConfigLoader.get_config()
    Embeddings.set_config_and_get_umls_mapper(config)

    news_path = os.path.join(config['PATH']['News'], 'news_2015_3M-sentences.txt')
    news_path_julie = os.path.join(config['PATH']['News'], 'news_2015_3M-sentences_JULIE.txt')
    ggponc_path = os.path.join(config['PATH']['GGPONC'], 'Plain Text', 'cpg-sentences.txt')
    ggponc_path_julie = os.path.join(config['PATH']['GGPONC'], 'Plain Text', 'cpg-sentences_JULIE.txt')
    jsynncc_path = os.path.join(config['PATH']['JSynnCC'], 'jsynncc-sentences.txt')
    jsynncc_path_julie = os.path.join(config['PATH']['JSynnCC'], 'jsynncc-sentences_JULIE.txt')
    pubmed_path = os.path.join(config['PATH']['PubMed'], 'all_sentences.txt')
    pubmed_path_julie = os.path.join(config['PATH']['PubMed'], 'all_sentences_JULIE.txt')
    if not DataHandler.path_exists(pubmed_path):
        DataHandler.read_files_and_save_sentences_to_dir(config['PATH']['PubMed'])
    paths = [
        ggponc_path,
        jsynncc_path,
        pubmed_path,
    ]
    paths_julie = [
        ggponc_path_julie,
        jsynncc_path_julie,
        pubmed_path_julie,
    ]

    # News
    # Embeddings.sentence_data2vec(path=news_path,
    #                              embedding_name="3M_news",
    #                              embeddings_algorithm="word2vec")
    Embeddings.sentence_data2vec(path=news_path,
                                 embedding_name="100K_news",
                                 embeddings_algorithm="word2vec",
                                 number_sentences=100000)
    # Embeddings.sentence_data2vec(path=news_path,
    #                              embedding_name="100K_news_fastText",
    #                              embeddings_algorithm="fastText",
    #                              number_sentences=100000)
    # Embeddings.sentence_data2vec(path=news_path,
    #                              embedding_name="100K_news_Glove",
    #                              embeddings_algorithm="Glove",
    #                              number_sentences=100000)
    # Embeddings.sentence_data2vec(path=news_path,
    #                              embedding_name="100K_news_plain",
    #                              embeddings_algorithm="word2vec",
    #                              number_sentences=100000,
    #                              use_multiterm_replacement=False)
    # Embeddings.sentence_data2vec(path=news_path,
    #                              embedding_name="100K_news_fastText_plain",
    #                              embeddings_algorithm="fastText",
    #                              number_sentences=100000,
    #                              use_multiterm_replacement=False)
    # Embeddings.sentence_data2vec(path=news_path,
    #                              embedding_name="100K_news_Glove_plain",
    #                              embeddings_algorithm="Glove",
    #                              number_sentences=100000,
    #                              use_multiterm_replacement=False)
    Embeddings.sentence_data2vec(path=news_path_julie,
                                 embedding_name="100K_news_JULIE",
                                 embeddings_algorithm="word2vec",
                                 number_sentences=100000,
                                 umls_replacement=False)
    # Embeddings.sentence_data2vec(path=news_path_julie,
    #                              embedding_name="100K_news_fastText_JULIE",
    #                              embeddings_algorithm="fastText",
    #                              number_sentences=100000,
    #                              umls_replacement=False)
    # Embeddings.sentence_data2vec(path=news_path_julie,
    #                              embedding_name="100K_news_Glove_JULIE",
    #                              embeddings_algorithm="Glove",
    #                              number_sentences=100000,
    #                              umls_replacement=False)
    # Embeddings.sentence_data2vec(path=news_path,
    #                              embedding_name="100K_news_no_cui",
    #                              embeddings_algorithm="word2vec",
    #                              number_sentences=100000,
    #                              umls_replacement=False)
    # Embeddings.sentence_data2vec(path=news_path,
    #                              embedding_name="100K_news_fastText_no_cui",
    #                              embeddings_algorithm="fastText",
    #                              number_sentences=100000,
    #                              umls_replacement=False)
    # Embeddings.sentence_data2vec(path=news_path,
    #                              embedding_name="100K_news_Glove_no_cui",
    #                              embeddings_algorithm="Glove",
    #                              number_sentences=100000,
    #                              umls_replacement=False)
    # Embeddings.sentence_data2vec(path=news_path,
    #                              embedding_name="3M_news_no_cui",
    #                              embeddings_algorithm="word2vec",
    #                              umls_replacement=False)
    # Embeddings.sentence_data2vec(path=news_path,
    #                              embedding_name="3M_news_fastText_no_cui",
    #                              embeddings_algorithm="fastText",
    #                              umls_replacement=False)
    # Embeddings.sentence_data2vec(path=news_path,
    #                              embedding_name="3M_news_Glove_no_cui",
    #                              embeddings_algorithm="Glove",
    #                              umls_replacement=False)

    # GGPONC
    # Embeddings.sentence_data2vec(path=ggponc_path,
    #                              embedding_name="GGPONC",
    #                              embeddings_algorithm="word2vec")
    #
    # Embeddings.sentence_data2vec(path=ggponc_path,
    #                              embedding_name="GGPONC_fastText",
    #                              embeddings_algorithm="fastText")
    # Embeddings.sentence_data2vec(path=ggponc_path,
    #                              embedding_name="GGPONC_glove",
    #                              embeddings_algorithm="Glove")
    # Embeddings.sentence_data2vec(path=ggponc_path,
    #                              embedding_name="GGPONC_plain",
    #                              embeddings_algorithm="word2vec",
    #                              use_multiterm_replacement=False
    #                              )
    # Embeddings.sentence_data2vec(path=ggponc_path,
    #                              embedding_name="GGPONC_fastText_plain",
    #                              embeddings_algorithm="fastText",
    #                              use_multiterm_replacement=False
    #                              )
    Embeddings.sentence_data2vec(path=ggponc_path,
                                 embedding_name="GGPONC_glove_plain",
                                 embeddings_algorithm="Glove",
                                 use_multiterm_replacement=False
                                 )
    # Embeddings.sentence_data2vec(path=ggponc_path,
    #                              embedding_name="GGPONC_no_cui",
    #                              embeddings_algorithm="word2vec",
    #                              umls_replacement=False
    #                              )
    # Embeddings.sentence_data2vec(path=ggponc_path,
    #                              embedding_name="GGPONC_fastText_no_cui",
    #                              embeddings_algorithm="fastText",
    #                              umls_replacement=False
    #                              )
    # Embeddings.sentence_data2vec(path=ggponc_path,
    #                              embedding_name="GGPONC_glove_no_cui",
    #                              embeddings_algorithm="Glove",
    #                              umls_replacement=False
    #                              )

    # Embeddings.sentence_data2vec(path=ggponc_path_julie,
    #                              embedding_name="GGPONC_JULIE",
    #                              embeddings_algorithm="word2vec",
    #                              umls_replacement=False)
    # Embeddings.sentence_data2vec(path=ggponc_path_julie,
    #                              embedding_name="GGPONC_fastText_JULIE",
    #                              embeddings_algorithm="fastText",
    #                              umls_replacement=False)
    # Embeddings.sentence_data2vec(path=ggponc_path_julie,
    #                              embedding_name="GGPONC_glove_JULIE",
    #                              embeddings_algorithm="Glove",
    #                              umls_replacement=False)
    #
    # # JSYNNCC
    # Embeddings.sentence_data2vec(path=jsynncc_path,
    #                              embedding_name="JSynnCC",
    #                              embeddings_algorithm="word2vec")
    # Embeddings.sentence_data2vec(path=jsynncc_path,
    #                              embedding_name="JSynnCC_fastText",
    #                              embeddings_algorithm="fastText")
    # Embeddings.sentence_data2vec(path=jsynncc_path,
    #                              embedding_name="JSynnCC_glove",
    #                              embeddings_algorithm="Glove")
    #
    # Embeddings.sentence_data2vec(path=jsynncc_path,
    #                              embedding_name="JSynnCC_plain",
    #                              embeddings_algorithm="word2vec",
    #                              use_multiterm_replacement=False
    #                              )
    # Embeddings.sentence_data2vec(path=jsynncc_path,
    #                              embedding_name="JSynnCC_fastText_plain",
    #                              embeddings_algorithm="fastText",
    #                              use_multiterm_replacement=False
    #                              )
    # Embeddings.sentence_data2vec(path=jsynncc_path,
    #                              embedding_name="JSynnCC_glove_plain",
    #                              embeddings_algorithm="Glove",
    #                              use_multiterm_replacement=False
    #                              )
    # Embeddings.sentence_data2vec(path=jsynncc_path,
    #                              embedding_name="JSynnCC_no_cui",
    #                              embeddings_algorithm="word2vec",
    #                              umls_replacement=False
    #                              )
    # Embeddings.sentence_data2vec(path=jsynncc_path,
    #                              embedding_name="JSynnCC_fastText_no_cui",
    #                              embeddings_algorithm="fastText",
    #                              umls_replacement=False
    #                              )
    # Embeddings.sentence_data2vec(path=jsynncc_path,
    #                              embedding_name="JSynnCC_glove_no_cui",
    #                              embeddings_algorithm="Glove",
    #                              umls_replacement=False
    #                              )
    # Embeddings.sentence_data2vec(path=jsynncc_path_julie,
    #                              embedding_name="JSynnCC_JULIE",
    #                              embeddings_algorithm="word2vec",
    #                              umls_replacement=False
    #                              )
    # Embeddings.sentence_data2vec(path=jsynncc_path_julie,
    #                              embedding_name="JSynnCC_fastText_JULIE",
    #                              embeddings_algorithm="fastText",
    #                              umls_replacement=False
    #                              )
    # Embeddings.sentence_data2vec(path=jsynncc_path_julie,
    #                              embedding_name="JSynnCC_glove_JULIE",
    #                              embeddings_algorithm="Glove",
    #                              umls_replacement=False
    #                              )
    # German PubMed
    # Embeddings.sentence_data2vec(path=pubmed_path,
    #                              embedding_name="PubMed",
    #                              embeddings_algorithm="word2vec")
    # Embeddings.sentence_data2vec(path=pubmed_path,
    #                              embedding_name="PubMed_fastText",
    #                              embeddings_algorithm="fastText")
    # Embeddings.sentence_data2vec(path=pubmed_path,
    #                              embedding_name="PubMed_glove",
    #                              embeddings_algorithm="Glove")
    # Embeddings.sentence_data2vec(path=pubmed_path,
    #                              embedding_name="PubMed_plain",
    #                              embeddings_algorithm="word2vec",
    #                              use_multiterm_replacement=False
    #                              )
    # Embeddings.sentence_data2vec(path=pubmed_path,
    #                              embedding_name="PubMed_fastText_plain",
    #                              embeddings_algorithm="fastText",
    #                              use_multiterm_replacement=False
    #                              )
    # Embeddings.sentence_data2vec(path=pubmed_path,
    #                              embedding_name="PubMed_glove_plain",
    #                              embeddings_algorithm="Glove",
    #                              use_multiterm_replacement=False
    #                              )
    # Embeddings.sentence_data2vec(path=pubmed_path,
    #                              embedding_name="PubMed_no_cui",
    #                              embeddings_algorithm="word2vec",
    #                              umls_replacement=False
    #                              )
    # Embeddings.sentence_data2vec(path=pubmed_path,
    #                              embedding_name="PubMed_fastText_no_cui",
    #                              embeddings_algorithm="fastText",
    #                              umls_replacement=False
    #                              )
    # Embeddings.sentence_data2vec(path=pubmed_path,
    #                              embedding_name="PubMed_glove_no_cui",
    #                              embeddings_algorithm="Glove",
    #                              umls_replacement=False
    #                              )
    # Embeddings.sentence_data2vec(path=pubmed_path_julie,
    #                              embedding_name="PubMed_JULIE",
    #                              embeddings_algorithm="word2vec",
    #                              umls_replacement=False
    #                              )
    # Embeddings.sentence_data2vec(path=pubmed_path_julie,
    #                              embedding_name="PubMed_fastText_JULIE",
    #                              embeddings_algorithm="fastText",
    #                              umls_replacement=False
    #                              )
    # Embeddings.sentence_data2vec(path=pubmed_path_julie,
    #                              embedding_name="PubMed_glove_JULIE",
    #                              embeddings_algorithm="Glove",
    #                              umls_replacement=False
    #                              )

    # Medical Concat
    Embeddings.sentence_data2vec(path=paths,
                                 embedding_name="German_Medical",
                                 embeddings_algorithm="word2vec",
                                 restrict_vectors=True,
                                 umls_replacement=True,
                                 use_multiterm_replacement=True
                                 )
    #
    # Embeddings.sentence_data2vec(path=paths,
    #                              embedding_name="German_Medical_plain",
    #                              embeddings_algorithm="word2vec",
    #                              umls_replacement=True,
    #                              use_multiterm_replacement=False
    #                              )
    #
    # Embeddings.sentence_data2vec(path=paths,
    #                              embedding_name="German_Medical_no_cui",
    #                              embeddings_algorithm="word2vec",
    #                              umls_replacement=False
    #                              )
    #
    # Embeddings.sentence_data2vec(path=paths,
    #                              embedding_name="German_Medical_fastText",
    #                              embeddings_algorithm="fastText",
    #                              umls_replacement=True,
    #                              use_multiterm_replacement=True
    #                              )
    #
    # Embeddings.sentence_data2vec(path=paths,
    #                              embedding_name="German_Medical_fastText_plain",
    #                              embeddings_algorithm="fastText",
    #                              umls_replacement=True,
    #                              use_multiterm_replacement=False
    #                             )
    #
    # Embeddings.sentence_data2vec(path=paths,
    #                              embedding_name="German_Medical_fastText_no_cui",
    #                              embeddings_algorithm="fastText",
    #                              umls_replacement=False
    #                              )
    #
    # Embeddings.sentence_data2vec(path=paths,
    #                              embedding_name="German_Medical_Glove",
    #                              embeddings_algorithm="Glove",
    #                              umls_replacement=True,
    #                              use_multiterm_replacement=True
    #                              )
    #
    # Embeddings.sentence_data2vec(path=paths,
    #                              embedding_name="German_Medical_Glove_plain",
    #                              embeddings_algorithm="Glove",
    #                              umls_replacement=True,
    #                              use_multiterm_replacement=False
    #                              )
    #
    # Embeddings.sentence_data2vec(path=paths,
    #                              embedding_name="German_Medical_Glove_no_cui",
    #                              embeddings_algorithm="Glove",
    #                              umls_replacement=False
    #                              )
    Embeddings.sentence_data2vec(path=paths_julie,
                                 embedding_name="German_Medical_JULIE",
                                 embeddings_algorithm="word2vec",
                                 umls_replacement=False
                                 )
    # Embeddings.sentence_data2vec(path=paths_julie,
    #                              embedding_name="German_Medical_fastText_JULIE",
    #                              embeddings_algorithm="fastText",
    #                              umls_replacement=False
    #                              )
    # Embeddings.sentence_data2vec(path=paths_julie,
    #                              embedding_name="German_Medical_Glove_JULIE_NEW",
    #                              embeddings_algorithm="Glove",
    #                              umls_replacement=False,
    #                              )
    # Flair
    # Embeddings.sentence_data2vec(path=ggponc_path,
    #                              embedding_name="Flair",
    #                              embeddings_algorithm="Flair",
    #                              flair_corpus_path="data/flair_test123",
    #                              flair_model_path='resources/taggers/language_model')


if __name__ == "__main__":
    main()
