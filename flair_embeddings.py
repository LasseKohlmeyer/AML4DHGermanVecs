from vectorization.embeddings import Flair, Embeddings
from utils.transform_data import DataHandler


if __name__ == '__main__':
    paths_to_input_sentences = [
        'E:/AML4DH-DATA/corp_test_sentences.txt'
    ]
    corpus_path = 'E:/AML4DH-DATA/test_corp'
    # flair_model_path = 'resources/taggers/language_model'
    flair_model_path = 'data/tes_fla'
    # flair_model_path = 'bert-base-german-cased'
    retrain_corpus_path = 'data/mini_test'
    save_path = f"E:/AML4DHGermanVecs/data/conv_flair_all.kv"
    sentences = DataHandler.concat_path_sentences(paths_to_input_sentences)[:10]
    vecs = Flair.get_flair_vectors(sentences, flair_model_path=flair_model_path, flair_algorithm='de-forward',
                                   retrain_corpus_path=None,
                                   epochs=1
                                   )

    Embeddings.save(vecs, path=save_path)
