import os
import warnings
from collections import defaultdict
from multiprocessing.spawn import freeze_support
from typing import List, Union
from tqdm import tqdm
from transform_data import DataHandler
import numpy as np
from embeddings import Embeddings
from sklearn.model_selection import train_test_split
warnings.simplefilter(action='ignore', category=FutureWarning)
from flair.data import Dictionary, Sentence
from flair.embeddings import FlairEmbeddings, TransformerWordEmbeddings
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus


def determine_algorithm_from_string(flair_algorithm_string: str):
    print(flair_algorithm_string)
    if 'bert' in flair_algorithm_string:
        return TransformerWordEmbeddings, 'bert'
    else:
        return FlairEmbeddings, 'flair'


def build_flair_corpus(sentences: List[str], root_path: str):
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    val_ratio = 0.2
    test_ratio = 0.2

    first_split = val_ratio + test_ratio
    second_split = test_ratio / first_split

    print('split data')
    train_sentences, val_sentences = train_test_split(sentences, test_size=first_split, random_state=42)
    val_sentences, test_sentences = train_test_split(val_sentences, test_size=second_split, random_state=42)

    train_splits = chunks(train_sentences, 5000)

    print('save splits')
    try:
        os.mkdir(root_path)
        os.mkdir(os.path.join(root_path, 'train'))
    except OSError:
        print("Creation of the directory %s failed" % root_path)

    for j, split in enumerate(train_splits):
        DataHandler.save(os.path.join(root_path, 'train', f'train_split_{j}.txt'), '\n'.join(split))
    DataHandler.save(os.path.join(root_path, 'valid.txt'), '\n'.join(val_sentences))
    DataHandler.save(os.path.join(root_path, 'test.txt'), '\n'.join(test_sentences))

    return sentences


def retrain_flair(corpus_path: str, model_path: str, flair_algorithm: str = 'de-forward'):
    use_embedding, algorithm = determine_algorithm_from_string(flair_algorithm_string=flair_algorithm)
    # instantiate an existing LM, such as one from the FlairEmbeddings
    model = use_embedding(flair_algorithm)
    if algorithm == 'bert':
        language_model = model.model
    else:
        language_model = model.lm

    # are you fine-tuning a forward or backward LM?
    is_forward_lm = language_model.is_forward_lm

    # get the dictionary from the existing language model
    dictionary: Dictionary = language_model.dictionary

    # get your corpus, process forward and at the character level
    corpus = TextCorpus(corpus_path,
                        dictionary,
                        is_forward_lm,
                        character_level=True)

    # use the model trainer to fine-tune this model on your corpus
    trainer = LanguageModelTrainer(language_model, corpus)

    trainer.train(model_path,
                  sequence_length=10,
                  mini_batch_size=10,
                  learning_rate=20,
                  max_epochs=1,
                  patience=10,
                  checkpoint=True)


def flair_embedding(raw_sentences: Union[List[str], List[List[str]]],
                    flair_model_path: str,
                    retrain_corpus_path: str = None,
                    flair_algorithm: str = None):
    freeze_support()
    if flair_algorithm is None:
        flair_algorithm = flair_model_path
    if retrain_corpus_path:
        if not os.path.isdir(retrain_corpus_path):
            raw_sentences = build_flair_corpus(raw_sentences, retrain_corpus_path)
        retrain_flair(retrain_corpus_path, flair_model_path, flair_algorithm=flair_algorithm)
    if os.path.exists(os.path.dirname(flair_model_path)):
        flair_model_path = os.path.join(flair_model_path, 'best-lm.pt')

    use_embedding, _ = determine_algorithm_from_string(flair_algorithm_string=flair_algorithm)
    print(use_embedding, flair_algorithm)
    embedding = use_embedding(flair_model_path)

    if any(isinstance(el, list) for el in raw_sentences):
        use_tokenizer = False
        raw_sentences = [' '.join(raw_sentence) for raw_sentence in raw_sentences if len(raw_sentence) > 0]
    else:
        use_tokenizer = True

    flair_sents = [Sentence(raw_sentence, use_tokenizer=use_tokenizer)
                   for raw_sentence in tqdm(raw_sentences,
                                            desc="Convert to flair",
                                            total=len(raw_sentences))]

    keyed_vecs = defaultdict(list)
    for flair_sentence in tqdm(flair_sents, desc='Embed sentences', total=len(flair_sents)):
        embedding.embed(flair_sentence)
        for token in flair_sentence:
            keyed_vecs[token.text].append(token.embedding)

    keyed_vecs = {key: np.array(sum(vecs) / len(vecs)) for key, vecs in keyed_vecs.items()}

    return Embeddings.to_gensim_binary(keyed_vecs)


# if __name__ == '__main__':
#     paths_to_input_sentences = [
#         'E:/AML4DH-DATA/corp_test_sentences.txt'
#     ]
#     corpus_path = 'E:/AML4DH-DATA/test_corp'
#     # flair_model_path = 'resources/taggers/language_model'
#     flair_model_path = 'de-forward'
#     # flair_model_path = 'bert-base-german-cased'
#     save_path = f"E:/AML4DHGermanVecs/data/conv_flair_all.kv"
#     sentences = DataHandler.concat_path_sentences(paths_to_input_sentences)[:10]
#     vecs = flair_embedding(sentences, flair_model_path, retrain_corpus_path=None)
#
#     Embeddings.save(vecs, path=save_path)
