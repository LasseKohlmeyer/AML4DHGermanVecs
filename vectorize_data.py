from typing import List, Union

import gensim
import spacy

from UMLS import UMLSMapper
from embeddings import Embeddings
from transform_data import DataHandler


def preprocess(tokens: List[str] = None, documents: List[List[str]] = None, lemmatize: bool = False, lower: bool = False,
               pos_filter: list = None, remove_stopwords: bool = False,
               remove_punctuation: bool = False, lan_model=None) -> List[List[str]]:
    def token_representation(token):
        representation = str(token.lemma_) if lemmatize else str(token)
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


def sentence_data2vec(path: str, embedding_name: str,
                      embeddings_algorithm: Union[str, gensim.models.Word2Vec, gensim.models.FastText] = "word2vec",
                      number_sentences: int = None,
                      use_phrases: bool = False,
                      restrict_vectors: bool = False):
    if isinstance(embeddings_algorithm, str) and embeddings_algorithm.lower() == "word2vec":
        embeddings_algorithm = gensim.models.Word2Vec
    if isinstance(embeddings_algorithm, str) and embeddings_algorithm.lower() == "fasttext":
        embeddings_algorithm = gensim.models.FastText
    if isinstance(embeddings_algorithm, str) and embeddings_algorithm.lower() == "glove":
        embeddings_algorithm = Embeddings.glove_vectors

    umls_mapper = UMLSMapper(from_dir='E:/AML4DH-DATA/UMLS')
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
    data_sentences = umls_mapper.replace_documents_with_spacy(data_sentences)
    print(data_sentences[:10])
    # for s in data_sentences:
    #     print(s)
    # data_sentences = umls_mapper.replace_documents_with_umls_smart(data_sentences)
    # data_sentences = preprocess(documents=data_sentences, lemmatize=True, remove_stopwords=True)


    # vecs = Embeddings.calculate_vectors([cpg_words], use_phrases=False)
    vecs = Embeddings.calculate_vectors(data_sentences, use_phrases=use_phrases, embedding_algorithm=embeddings_algorithm)
    print(f'Got {len(vecs.vocab)} vectors for {len(data_sentences)} sentences')

    Embeddings.save_medical(vecs, embedding_name, umls_mapper, restrict=restrict_vectors)

def main():

    # News
    # sentence_data2vec(path="E:/AML4DH-DATA/2015_3M_sentences/deu_news_2015_3M-sentences.txt",
    #                   embedding_name="3M_news",
    #                   embeddings_algorithm="word2vec",
    #                   number_sentences=None)

    # sentence_data2vec(path="E:/AML4DH-DATA/2015_3M_sentences/deu_news_2015_3M-sentences.txt",
    #                   embedding_name="60K_news_fastText",
    #                   embeddings_algorithm="fastText",
    #                   number_sentences=60000)

    # sentence_data2vec(path="E:/AML4DH-DATA/2015_3M_sentences/deu_news_2015_3M-sentences.txt",
    #                   embedding_name="60K_news_Glove",
    #                   embeddings_algorithm="Glove",
    #                   number_sentences=60000)

    # GGPONC
    # sentence_data2vec(path="E:/AML4DH-DATA/CPG-AMIA2020/Plain Text/cpg-sentences.txt",
    #                   embedding_name="GGPONC",
    #                   embeddings_algorithm="word2vec")

    # sentence_data2vec(path="E:/AML4DH-DATA/CPG-AMIA2020/Plain Text/cpg-sentences.txt",
    #                   embedding_name="GGPONC_fastText",
    #                   embeddings_algorithm="fastText")

    # sentence_data2vec(path="E:/AML4DH-DATA/CPG-AMIA2020/Plain Text/cpg-sentences.txt",
    #                   embedding_name="GGPONC_glove",
    #                   embeddings_algorithm="Glove")

    # JSYNC
    # sentence_data2vec(path="E:/AML4DH-DATA/JSynCC/jsynncc-sentences.txt",
    #                   embedding_name="JSynCC",
    #                   embeddings_algorithm="word2vec")

    # German PubMed
    path = "E:/AML4DH-DATA/german_pubmed/all_sentences.txt"
    if not DataHandler.path_exists(path):
        DataHandler.read_files_and_save_sentences_to_dir("E:\AML4DH-DATA\german_pubmed")

    sentence_data2vec(path=path,
                      embedding_name="PubMed",
                      embeddings_algorithm="word2vec")


    # # load sentences
    # # https: // www.kaggle.com / rtatman / 3 - million - german - sentences / data?select = deu_news_2015_3M - sentences.txt
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
