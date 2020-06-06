import os
from typing import List

import spacy
from tqdm import tqdm
import pandas as pd



class DataHandler:
    @staticmethod
    def path_exists(path: str) -> bool:
        return os.path.isfile(path)

    @staticmethod
    def lines_from_file(path: str, encoding="utf-8") -> List[str]:
        with open(path, encoding=encoding) as f:
            data = f.read()

        return data.split("\n")

    @staticmethod
    def load_folder_textfiles(directory: str) -> str:
        files = []
        for (dirpath, dirnames, filenames) in os.walk(directory):
            files.extend(filenames)
            break

        for f in files:
            if f.startswith('.') or not f[-4:] == '.txt':
                files.remove(f)

        docs = []
        for file in tqdm(files, desc=f"load documents in {directory}", total=len(files)):
            docs.append(" ".join(DataHandler.lines_from_file(os.path.join(directory, file), encoding=None)).strip())
        return " ".join(docs)

    @staticmethod
    def sentenize(document_text):
        german_model = spacy.load("de_core_news_sm")
        sbd = german_model.create_pipe('sentencizer')
        german_model.add_pipe(sbd)

        doc = german_model(document_text)

        sents_list = []
        for sent in doc.sents:
            sents_list.append(sent.text)

        return sents_list

    @staticmethod
    def save(file_path, content):
        with open(file_path, 'w', encoding="utf-8") as the_file:
            the_file.write(content)

    @staticmethod
    def read_files_and_save_sentences_to_dir(directory: str):
        sentences = DataHandler.sentenize(DataHandler.load_folder_textfiles(directory))
        DataHandler.save(file_path=os.path.join(directory, "all_sentences.txt"), content="\n".join(sentences))


# DataHandler.read_files_and_save_sentences_to_dir("E:\AML4DH-DATA\german_pubmed")