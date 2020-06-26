import os
import warnings
from multiprocessing.spawn import freeze_support
from typing import List

from transform_data import DataHandler

warnings.simplefilter(action='ignore', category=FutureWarning)

from flair.data import Dictionary, Sentence
from flair.embeddings import FlairEmbeddings, TransformerWordEmbeddings
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus
from sklearn.model_selection import train_test_split


def build_flair_corpus(paths: List[str], root_path: str):
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    sentences = []
    val_ratio = 0.2
    test_ratio = 0.2

    print('read data')
    for path in paths:
        sentences.extend(DataHandler.lines_from_file(path))

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


def retrain_flair(corpus_path: str, model_path: str = 'resources/taggers/language_model'):
    # instantiate an existing LM, such as one from the FlairEmbeddings
    language_model = FlairEmbeddings('news-forward').lm
    language_model = FlairEmbeddings('de-forward').lm
    # language_model = TransformerWordEmbeddings('bert-base-german-cased').model

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

    trainer.train('resources/taggers/language_model',
                  sequence_length=10,
                  mini_batch_size=10,
                  learning_rate=20,
                  max_epochs=1,
                  patience=10,
                  checkpoint=True)


if __name__ == '__main__':
    freeze_support()

    # paths_to_input_sentences = [
    #     'E:/AML4DH-DATA/corp_test_sentences.txt'
    # ]
    # corpus_path = 'E:/AML4DH-DATA/test_corp'
    #
    # if not os.path.isdir(corpus_path):
    #     build_flair_corpus(paths_to_input_sentences, corpus_path)
    # retrain_flair(corpus_path)

    emb = FlairEmbeddings('resources/taggers/language_model/best-lm.pt')
    sentence = Sentence('Das  ist grün .')

    # embed a sentence using glove.
    emb.embed(sentence)

    for token in sentence:
        print(token)
        print(token.embedding)