from german_vec_pipeline.embeddings import Embeddings
from german_vec_pipeline.transform_data import DataHandler


def main():
    # News
    # sentence_data2vec(path="E:/AML4DH-DATA/2015_3M_sentences/news_2015_3M-sentences.txt",
    #                   embedding_name="3M_news",
    #                   embeddings_algorithm="word2vec",
    #                   number_sentences=None)

    # sentence_data2vec(path="E:/AML4DH-DATA/2015_3M_sentences/news_2015_3M-sentences.txt",
    #                   embedding_name="60K_news_fastText",
    #                   embeddings_algorithm="fastText",
    #                   number_sentences=60000)

    # sentence_data2vec(path="E:/AML4DH-DATA/2015_3M_sentences/news_2015_3M-sentences.txt",
    #                   embedding_name="60K_news_Glove",
    #                   embeddings_algorithm="Glove",
    #                   number_sentences=60000)
    # sentence_data2vec(path="E:/AML4DH-DATA/2015_3M_sentences/news_2015_3M-sentences_JULIE.txt",
    #                   embedding_name="60K_news_JULIE",
    #                   embeddings_algorithm="word2vec",
    #                   number_sentences=60000,
    #                   umls_replacement=False)
    # sentence_data2vec(path="E:/AML4DH-DATA/2015_3M_sentences/news_2015_3M-sentences.txt",
    #                   embedding_name="60K_news_no_cui",
    #                   embeddings_algorithm="word2vec",
    #                   number_sentences=60000,
    #                   umls_replacement=False)
    # sentence_data2vec(path="E:/AML4DH-DATA/2015_3M_sentences/news_2015_3M-sentences.txt",
    #                   embedding_name="60K_news_plain",
    #                   embeddings_algorithm="word2vec",
    #                   number_sentences=60000,
    #                   use_multiterm_replacement=False)

    # GGPONC
    # sentence_data2vec(path="E:/AML4DH-DATA/CPG-AMIA2020/Plain Text/cpg-sentences.txt",
    #                   embedding_name="GGPONC",
    #                   embeddings_algorithm="word2vec")

    # sentence_data2vec(path="E:/AML4DH-DATA/CPG-AMIA2020/Plain Text/cpg-sentences.txt",
    #                   embedding_name="GGPONC_fastText",
    #                   embeddings_algorithm="fastText")
    # #fixme check: still nan error?
    # sentence_data2vec(path="E:/AML4DH-DATA/CPG-AMIA2020/Plain Text/cpg-sentences.txt",
    #                   embedding_name="GGPONC_glove",
    #                   embeddings_algorithm="Glove")
    # sentence_data2vec(path="E:/AML4DH-DATA/CPG-AMIA2020/Plain Text/cpg-sentences_JULIE.txt",
    #                   embedding_name="GGPONC_JULIE",
    #                   embeddings_algorithm="word2vec",
    #                   umls_replacement=False
    #                   )
    # sentence_data2vec(path="E:/AML4DH-DATA/CPG-AMIA2020/Plain Text/cpg-sentences.txt",
    #                   embedding_name="GGPONC_no_cui",
    #                   embeddings_algorithm="word2vec",
    #                   umls_replacement=False
    #                   )
    # sentence_data2vec(path="E:/AML4DH-DATA/CPG-AMIA2020/Plain Text/cpg-sentences.txt",
    #                   embedding_name="GGPONC_plain",
    #                   embeddings_algorithm="word2vec",
    #                   use_multiterm_replacement=False
    #                   )

    # JSYNC
    # sentence_data2vec(path="E:/AML4DH-DATA/JSynCC/jsynncc-sentences.txt",
    #                   embedding_name="JSynCC",
    #                   embeddings_algorithm="word2vec")

    # German PubMed
    path = "E:/AML4DH-DATA/german_pubmed/all_sentences.txt"
    if not DataHandler.path_exists(path):
        DataHandler.read_files_and_save_sentences_to_dir("E:\AML4DH-DATA\german_pubmed")

    Embeddings.sentence_data2vec(path=path,
                                 embedding_name="PubMed",
                                 embeddings_algorithm="word2vec")

    # Medical Concat
    # paths = [
    #     "E:/AML4DH-DATA/CPG-AMIA2020/Plain Text/cpg-sentences.txt",
    #     "E:/AML4DH-DATA/JSynCC/jsynncc-sentences.txt",
    #     "E:/AML4DH-DATA/german_pubmed/all_sentences.txt"
    # ]
    # Embeddings.sentence_data2vec(path=paths,
    #                   embedding_name="German_Medical",
    #                   embeddings_algorithm="word2vec",
    #                   restrict_vectors=True,
    #                   umls_replacement=True,
    #                   use_multiterm_replacement=True
    #                   )
    #
    # Embeddings.sentence_data2vec(path=paths,
    #                   embedding_name="German_Medical_plain",
    #                   embeddings_algorithm="word2vec",
    #                   umls_replacement=True,
    #                   use_multiterm_replacement=False
    #                   )
    #
    # Embeddings.sentence_data2vec(path=paths,
    #                   embedding_name="German_Medical_no_cui",
    #                   embeddings_algorithm="word2vec",
    #                   umls_replacement=False
    #                   )

    # Embeddings.sentence_data2vec(path=paths,
    #                   embedding_name="German_Medical_fastText",
    #                   embeddings_algorithm="fastText",
    #                   umls_replacement=True,
    #                   use_multiterm_replacement=True
    #                   )
    #
    # Embeddings.sentence_data2vec(path=paths,
    #                   embedding_name="German_Medical_fastText_plain",
    #                   embeddings_algorithm="fastText",
    #                   umls_replacement=True,
    #                   use_multiterm_replacement=False
    #                   )
    #
    # Embeddings.sentence_data2vec(path=paths,
    #                   embedding_name="German_Medical_fastText_no_cui",
    #                   embeddings_algorithm="fastText",
    #                   umls_replacement=False
    #                   )

    # Embeddings.sentence_data2vec(path=paths,
    #                   embedding_name="German_Medical_Glove",
    #                   embeddings_algorithm="Glove",
    #                   umls_replacement=True,
    #                   use_multiterm_replacement=True
    #                   )
    #
    # Embeddings.sentence_data2vec(path=paths,
    #                   embedding_name="German_Medical_Glove_plain",
    #                   embeddings_algorithm="Glove",
    #                   umls_replacement=True,
    #                   use_multiterm_replacement=False
    #                   )
    #
    # Embeddings.sentence_data2vec(path=paths,
    #                   embedding_name="German_Medical_Glove_no_cui",
    #                   embeddings_algorithm="Glove",
    #                   umls_replacement=False
    #                   )

    # paths = [
    #     "E:/AML4DH-DATA/CPG-AMIA2020/Plain Text/cpg-sentences_JULIE.txt",
    #     "E:/AML4DH-DATA/JSynCC/jsynncc-sentences_JULIE.txt",
    #     "E:/AML4DH-DATA/german_pubmed/all_sentences_JULIE.txt"
    # ]
    # Embeddings.sentence_data2vec(path=paths,
    #                   embedding_name="German_Medical_JULIE",
    #                   embeddings_algorithm="word2vec",
    #                   umls_replacement=False
    #                   )
    # Embeddings.sentence_data2vec(path=paths,
    #                   embedding_name="German_Medical_fastText_JULIE",
    #                   embeddings_algorithm="fastText",
    #                   umls_replacement=False
    #                   )
    # Embeddings.sentence_data2vec(path=paths,
    #                   embedding_name="German_Medical_Glove_JULIE",
    #                   embeddings_algorithm="Glove",
    #                   umls_replacement=False
    #                   )
    # Flair
    Embeddings.sentence_data2vec(path="E:/AML4DH-DATA/CPG-AMIA2020/Plain Text/cpg-sentences.txt",
                      embedding_name="Flair",
                      embeddings_algorithm="Flair",
                      flair_corpus_path=None,
                      flair_model_path='resources/taggers/language_model')



    # # load sentences
    # # https://www.kaggle.com/rtatman/3-million-german-sentences/data?select=deu_news_2015_3M-sentences.txt
    # data_sentences = lines_from_file(path="E:/AML4DH-DATA/2015_3M_sentences/deu_news_2015_3M-sentences.txt")
    # # data_sentences = lines_from_file(path="E:/AML4DH-DATA/CPG-AMIA2020/Plain Text/cpg-sentences.txt")
    # print((data_sentences[:10]))
    #
    # # cpg_words = lines_from_file(path="E:/AML4DH-DATA/CPG-AMIA2020/Plain Text/cpg-tokens.txt")
    #
    # # Preprocessing
    # cpg_words = umls_mapper.standardize_words(cpg_words)
    # cpg_words = umls_mapper.replace_with_umls(cpg_words)
    # cpg_words = DataHandler.preprocess(cpg_words)
    # print((cpg_words[:100]))
    #
    # # data_sentences = umls_mapper.standardize_documents(data_sentences)
    # # data_sentences = umls_mapper.replace_documents_with_umls(data_sentences)
    # data_sentences = umls_mapper.replace_documents_with_spacy(data_sentences)
    # # data_sentences = umls_mapper.replace_documents_with_umls_smart(data_sentences)
    # # data_sentences = DataHandler.preprocess(documents=data_sentences, lemmatize=True, remove_stopwords=True)
    # print((data_sentences[:10]))
    #
    #
    # #
    # # # Vectorization
    # # vecs = Embeddings.calculate_vectors([cpg_words], use_phrases=False)
    # vecs = Embeddings.calculate_vectors(data_sentences, use_phrases=False)
    # file_name = "no_prep_vecs_test"
    #
    # Embeddings.save_medical(vecs, file_name, umls_mapper)



if __name__ == "__main__":
    main()
