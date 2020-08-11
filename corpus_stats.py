import os
from typing import List, Union

from utils.transform_data import DataHandler, ConfigLoader


def get_counts(paths=Union[List[str], str], number_sentences: int = None):
    if isinstance(paths, list):
        data_sentences = DataHandler.concat_path_sentences(paths)
    else:
        data_sentences = DataHandler.lines_from_file(path=paths)
    if number_sentences:
        data_sentences = data_sentences[:number_sentences]

    nr_sentences = len(data_sentences)
    tokens = [token for sentence in data_sentences for token in sentence.split()]
    nr_tokens = len(tokens)
    nr_types = len(set(tokens))
    return nr_tokens, nr_sentences, nr_types


def main():
    config = ConfigLoader.get_config()

    news_path = os.path.join(config['PATH']['News'], 'news_2015_3M-sentences.txt')
    # news_path_julie = os.path.join(config['PATH']['News'], 'news_2015_3M-sentences_JULIE.txt')
    ggponc_path = os.path.join(config['PATH']['GGPONC'], 'Plain Text', 'cpg-sentences.txt')
    # ggponc_path_julie = os.path.join(config['PATH']['GGPONC'], 'Plain Text', 'cpg-sentences_JULIE.txt')
    jsynncc_path = os.path.join(config['PATH']['JSynnCC'], 'jsynncc-sentences.txt')
    # jsynncc_path_julie = os.path.join(config['PATH']['JSynnCC'], 'jsynncc-sentences_JULIE.txt')
    pubmed_path = os.path.join(config['PATH']['PubMed'], 'all_sentences.txt')
    # pubmed_path_julie = os.path.join(config['PATH']['PubMed'], 'all_sentences_JULIE.txt')
    if not DataHandler.path_exists(pubmed_path):
        DataHandler.read_files_and_save_sentences_to_dir(config['PATH']['PubMed'])
    paths = [
        ggponc_path,
        jsynncc_path,
        pubmed_path,
    ]
    # paths_julie = [
    #     ggponc_path_julie,
    #     jsynncc_path_julie,
    #     pubmed_path_julie,
    # ]

    print(get_counts(ggponc_path))
    print(get_counts(jsynncc_path))
    print(get_counts(pubmed_path))
    print(get_counts(paths))
    print(get_counts(news_path))
    print(get_counts(news_path, number_sentences=100000))


if __name__ == "__main__":
    main()
