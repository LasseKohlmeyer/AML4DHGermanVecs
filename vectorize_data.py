from typing import List, Union

import gensim
import spacy

from UMLS import UMLSMapper
from embeddings import Embeddings
from train_flair_embeddings import flair_embedding
from transform_data import DataHandler


def preprocess(tokens: List[str] = None, documents: List[List[str]] = None, lemmatize: bool = False,
               lower: bool = False, pos_filter: list = None, remove_stopwords: bool = False,
               remove_punctuation: bool = False, lan_model=None) -> List[List[str]]:
    def token_representation(tok):
        representation = str(tok.lemma_) if lemmatize else str(tok)
        if lower:
            representation = representation.lower()
        return representation

    nlp = spacy.load("de_core_news_sm") if lan_model is None else lan_model
    nlp.Defaults.stop_words |= {"der", "die", "das", "Der", "Die", "Das", "bei", "Bei", "In", "in"}

    if tokens:
        for word in nlp.Defaults.stop_words:
            lex = nlp.vocab[word]
            lex.is_stop = True
        preprocessed_tokens = []

        if pos_filter is None:
            for doc in nlp.pipe(tokens, disable=['parser', 'ner', 'tagger']):
                for token in doc:
                    if (not remove_stopwords or not token.is_stop) and (not remove_punctuation or token.is_alpha):
                        preprocessed_tokens.append(token_representation(token))

        else:
            for doc in nlp.pipe(tokens, disable=['parser', 'ner']):
                for token in doc:
                    if (not remove_stopwords or not token.is_stop) and (
                            not remove_punctuation or token.is_alpha) and token.pos_ in pos_filter:
                        preprocessed_tokens.append(token_representation(token))
        return [preprocessed_tokens]

    if documents:
        documents = [' '.join(doc) for doc in documents]
        for word in nlp.Defaults.stop_words:
            lex = nlp.vocab[word]
            lex.is_stop = True
        preprocessed_documents = []

        if pos_filter is None:
            for doc in nlp.pipe(documents, disable=['parser', 'ner', 'tagger']):
                preprocessed_document = []
                for token in doc:
                    if (not remove_stopwords or not token.is_stop) and (not remove_punctuation or token.is_alpha):
                        preprocessed_document.append(token_representation(token))
                preprocessed_documents.append(preprocessed_document)

        else:
            for doc in nlp.pipe(documents, disable=['parser', 'ner']):
                preprocessed_document = []
                for token in doc:
                    if (not remove_stopwords or not token.is_stop) and (
                            not remove_punctuation or token.is_alpha) and token.pos_ in pos_filter:
                        preprocessed_document.append(token_representation(token))
                preprocessed_documents.append(preprocessed_document)

        return preprocessed_documents


def sentence_data2vec(path: Union[str, List[str]], embedding_name: str,
                      embeddings_algorithm: Union[str, gensim.models.Word2Vec, gensim.models.FastText] = "word2vec",
                      number_sentences: int = 1000,
                      use_phrases: bool = False,
                      restrict_vectors: bool = False,
                      umls_replacement: bool = True,
                      use_multiterm_replacement: bool = True,
                      flair_model_path: str = None,
                      flair_corpus_path: str = None,
                      ):
    is_flair = False
    if isinstance(embeddings_algorithm, str) and embeddings_algorithm.lower() == "word2vec":
        embeddings_algorithm = gensim.models.Word2Vec
    if isinstance(embeddings_algorithm, str) and embeddings_algorithm.lower() == "fasttext":
        embeddings_algorithm = gensim.models.FastText
    if isinstance(embeddings_algorithm, str) and embeddings_algorithm.lower() == "glove":
        embeddings_algorithm = Embeddings.glove_vectors
    if isinstance(embeddings_algorithm, str) and embeddings_algorithm.lower() == "flair":
        is_flair = True

    umls_mapper = UMLSMapper(from_dir='E:/AML4DH-DATA/UMLS')
    if isinstance(path, list):
        data_sentences = DataHandler.concat_path_sentences(path)
        # old:
        # data_sentences = []
        # for p in path:
        #     data_sentences.extend(DataHandler.lines_from_file(path=p))
    else:
        data_sentences = DataHandler.lines_from_file(path=path)
    if number_sentences:
        data_sentences = data_sentences[:number_sentences]
    print((data_sentences[:10]))


    # cpg_words = lines_from_file(path="E:/AML4DH-DATA/CPG-AMIA2020/Plain Text/cpg-tokens.txt")

    # Preprocessing
    # cpg_words = umls_mapper.standardize_words(cpg_words)
    # cpg_words = umls_mapper.replace_with_umls(cpg_words)
    # cpg_words = preprocess(cpg_words)
    # print((cpg_words[:100]))

    # data_sentences = umls_mapper.standardize_documents(data_sentences)
    # data_sentences = umls_mapper.replace_documents_with_umls(data_sentences)
    # sents = [data_sentences[20124], data_sentences[20139]]
    if umls_replacement:
        if use_multiterm_replacement:
            data_sentences = umls_mapper.replace_documents_with_spacy_multiterm(data_sentences)
        else:
            data_sentences = umls_mapper.replace_documents_token_based(data_sentences)
        print(data_sentences[:10])
    else:
        data_sentences = umls_mapper.spacy_tokenize(data_sentences)
        print(data_sentences[:10])
    # for s in data_sentences:
    #     print(s)
    # data_sentences = umls_mapper.replace_documents_with_umls_smart(data_sentences)
    # data_sentences = preprocess(documents=data_sentences, lemmatize=True, remove_stopwords=True)

    # vecs = Embeddings.calculate_vectors([cpg_words], use_phrases=False)

    if is_flair:
        vecs = flair_embedding(data_sentences, flair_model_path, flair_corpus_path)
    else:
        vecs = Embeddings.calculate_vectors(data_sentences,
                                            use_phrases=use_phrases,
                                            embedding_algorithm=embeddings_algorithm)

    print(f'Got {len(vecs.vocab)} vectors for {len(data_sentences)} sentences')

    Embeddings.save_medical(vecs, embedding_name, umls_mapper, restrict=restrict_vectors)


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
    # path = "E:/AML4DH-DATA/german_pubmed/all_sentences.txt"
    # if not DataHandler.path_exists(path):
    #     DataHandler.read_files_and_save_sentences_to_dir("E:\AML4DH-DATA\german_pubmed")
    #
    # sentence_data2vec(path=path,
    #                   embedding_name="PubMed",
    #                   embeddings_algorithm="word2vec")

    # Medical Concat
    # paths = [
    #     "E:/AML4DH-DATA/CPG-AMIA2020/Plain Text/cpg-sentences.txt",
    #     "E:/AML4DH-DATA/JSynCC/jsynncc-sentences.txt",
    #     "E:/AML4DH-DATA/german_pubmed/all_sentences.txt"
    # ]
    # sentence_data2vec(path=paths,
    #                   embedding_name="German_Medical",
    #                   embeddings_algorithm="word2vec",
    #                   restrict_vectors=True,
    #                   umls_replacement=True,
    #                   use_multiterm_replacement=True
    #                   )
    #
    # sentence_data2vec(path=paths,
    #                   embedding_name="German_Medical_plain",
    #                   embeddings_algorithm="word2vec",
    #                   umls_replacement=True,
    #                   use_multiterm_replacement=False
    #                   )
    #
    # sentence_data2vec(path=paths,
    #                   embedding_name="German_Medical_no_cui",
    #                   embeddings_algorithm="word2vec",
    #                   umls_replacement=False
    #                   )

    # sentence_data2vec(path=paths,
    #                   embedding_name="German_Medical_fastText",
    #                   embeddings_algorithm="fastText",
    #                   umls_replacement=True,
    #                   use_multiterm_replacement=True
    #                   )
    #
    # sentence_data2vec(path=paths,
    #                   embedding_name="German_Medical_fastText_plain",
    #                   embeddings_algorithm="fastText",
    #                   umls_replacement=True,
    #                   use_multiterm_replacement=False
    #                   )
    #
    # sentence_data2vec(path=paths,
    #                   embedding_name="German_Medical_fastText_no_cui",
    #                   embeddings_algorithm="fastText",
    #                   umls_replacement=False
    #                   )

    # sentence_data2vec(path=paths,
    #                   embedding_name="German_Medical_Glove",
    #                   embeddings_algorithm="Glove",
    #                   umls_replacement=True,
    #                   use_multiterm_replacement=True
    #                   )
    #
    # sentence_data2vec(path=paths,
    #                   embedding_name="German_Medical_Glove_plain",
    #                   embeddings_algorithm="Glove",
    #                   umls_replacement=True,
    #                   use_multiterm_replacement=False
    #                   )
    #
    # sentence_data2vec(path=paths,
    #                   embedding_name="German_Medical_Glove_no_cui",
    #                   embeddings_algorithm="Glove",
    #                   umls_replacement=False
    #                   )

    # paths = [
    #     "E:/AML4DH-DATA/CPG-AMIA2020/Plain Text/cpg-sentences_JULIE.txt",
    #     "E:/AML4DH-DATA/JSynCC/jsynncc-sentences_JULIE.txt",
    #     "E:/AML4DH-DATA/german_pubmed/all_sentences_JULIE.txt"
    # ]
    # sentence_data2vec(path=paths,
    #                   embedding_name="German_Medical_JULIE",
    #                   embeddings_algorithm="word2vec",
    #                   umls_replacement=False
    #                   )
    # sentence_data2vec(path=paths,
    #                   embedding_name="German_Medical_fastText_JULIE",
    #                   embeddings_algorithm="fastText",
    #                   umls_replacement=False
    #                   )
    # sentence_data2vec(path=paths,
    #                   embedding_name="German_Medical_Glove_JULIE",
    #                   embeddings_algorithm="Glove",
    #                   umls_replacement=False
    #                   )
    # Flair
    sentence_data2vec(path="E:/AML4DH-DATA/CPG-AMIA2020/Plain Text/cpg-sentences.txt",
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
    # # cpg_words = umls_mapper.standardize_words(cpg_words)
    # # cpg_words = umls_mapper.replace_with_umls(cpg_words)
    # # cpg_words = preprocess(cpg_words)
    # # print((cpg_words[:100]))
    #
    # # data_sentences = umls_mapper.standardize_documents(data_sentences)
    # # data_sentences = umls_mapper.replace_documents_with_umls(data_sentences)
    # data_sentences = umls_mapper.replace_documents_with_spacy(data_sentences)
    # # data_sentences = umls_mapper.replace_documents_with_umls_smart(data_sentences)
    # # data_sentences = preprocess(documents=data_sentences, lemmatize=True, remove_stopwords=True)
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
