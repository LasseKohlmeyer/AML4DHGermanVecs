from typing import List

import spacy

from UMLS import UMLSMapper
from embeddings import Embeddings


def lines_from_file(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        data = f.read()

    return data.split("\n")


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


def main():
    umls_mapper = UMLSMapper(from_dir='E:/AML4DH-DATA/UMLS')
    # load sentences
    cpq_sentences = lines_from_file(path="E:/AML4DH-DATA/CPG-AMIA2020/Plain Text/cpg-sentences.txt")

    # tokenization
    cpq_sentences = [sentence.split() for sentence in cpq_sentences]

    # cpg_words = lines_from_file(path="E:/AML4DH-DATA/CPG-AMIA2020/Plain Text/cpg-tokens.txt")

    # Preprocessing
    # cpg_words = umls_mapper.standardize_words(cpg_words)
    # cpg_words = umls_mapper.replace_with_umls(cpg_words)
    # cpg_words = preprocess(cpg_words)
    # print((cpg_words[:100]))

    # cpq_sentences = umls_mapper.standardize_documents(cpq_sentences)
    cpq_sentences = umls_mapper.replace_documents_with_umls(cpq_sentences)
    # cpq_sentences = preprocess(documents=cpq_sentences, lemmatize=True, remove_stopwords=True)
    print((cpq_sentences[:10]))


    #
    # # Vectorization
    # vecs = Embeddings.calculate_vectors([cpg_words], use_phrases=False)
    vecs = Embeddings.calculate_vectors(cpq_sentences, use_phrases=False)
    file_name = "no_prep_vecs"
    Embeddings.save(vecs, path=f"E:/AML4DHGermanVecs/{file_name}_all.kv")

    concept_vecs = umls_mapper.get_umls_vectors_only(vecs)
    Embeddings.restrict_vectors(vecs, concept_vecs.keys())
    Embeddings.save(vecs, path=f"E:/AML4DHGermanVecs/{file_name}.kv")


if __name__ == "__main__":
    main()
