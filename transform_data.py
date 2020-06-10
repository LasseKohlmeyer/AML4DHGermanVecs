import os
from collections import defaultdict, Counter
from typing import List, Dict

import spacy
from spacy.matcher.phrasematcher import PhraseMatcher
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

    @staticmethod
    def split_data(file_path: str, lines_per_file: int, new_name: str):
        lines = DataHandler.lines_from_file(file_path)
        total_lines = len(lines)
        i = 0
        while True:
            if i*lines_per_file > total_lines:
                break
            DataHandler.save(f'{new_name}_{i}.txt', "\n".join(lines[i*lines_per_file: (i+1)*lines_per_file]))
            i += 1

    @staticmethod
    def tidy_indices(file_path, new_path):
        lines = DataHandler.lines_from_file(file_path)
        lines = ["\t".join(line.split('\t')[1:]) for line in lines]
        DataHandler.save(new_path, "\n".join(lines))

    @staticmethod
    def replace_documents_with_spacy(documents: List[str], replacement_dict: Dict[str, str]) -> List[str]:
        nlp = spacy.load('de_core_news_sm')
        matcher = PhraseMatcher(nlp.vocab)
        terms = replacement_dict.keys()
        doc_pipe = list(nlp.pipe(documents, disable=["tagger", "parser", "ner"]))
        # Only run nlp.make_doc to speed things up
        patterns = [nlp.make_doc(term) for term in terms]
        matcher.add("TerminologyList", None, *patterns)
        replaced_docs = []
        replaced_cuis = []
        for doc in tqdm(doc_pipe, desc="Replace with concepts", total=len(documents)):
            text_doc = doc.text
            matches = matcher(doc)
            concepts = []
            for match_id, start, end in matches:
                span = doc[start:end]
                concepts.append(span.text)

            concepts.sort(key=lambda s: len(s), reverse=True)
            for concept in concepts:
                if True or concept in replacement_dict:
                    text_doc = text_doc.replace(concept, replacement_dict[concept])
                    replaced_cuis.append(replacement_dict[concept])

            replaced_docs.append(text_doc)

            # tokens = [token for token in text_doc.split()]
            # replaced_docs.append(tokens)
        # print(len(replaced_cuis), len(set(replaced_cuis)), replaced_cuis[:3])
        return replaced_docs

    @staticmethod
    def julielab_replacements(file_path, offset_path, new_path):
        def replace(string: str, replacement_dict: Dict[str, str]) -> str:
            for key, value in replacement_dict.items():
                if str(key) in str(string):
                    string = string.replace(f'{key}', f' {value} ')
            return string

        def most_common(input_list):
            occurence_count = Counter(input_list)
            return occurence_count.most_common(1)[0][0]

        sentences = DataHandler.lines_from_file(file_path)
        replacement_df = pd.read_csv(offset_path, delimiter="\t", header=None)
        replacement_df.columns = ['source', 'start', 'end', 'term', 'cui']

        replacements = defaultdict(list)
        for i, row in tqdm(replacement_df.iterrows(), total=len(replacement_df)):
            replacements[row['term']].append(row['cui'])

        if "." in replacements:
            del replacements["."]

        replacements = {str(k): str(most_common(vs)) for k, vs in replacements.items()}
        # print(len(replacements.keys()))
        # replaced_sentences = [replace(sentence, replacements) for sentence in tqdm(sentences)]

        replaced_sentences = DataHandler.replace_documents_with_spacy(sentences, replacements)
        # DataHandler.save(new_path, "\n".join(replaced_sentences))

    @staticmethod
    def julielab_replacements_offset_using(file_path, offset_path, new_path):
        def replace(string: str, replacement_dict: Dict[str, str]) -> str:
            for key, value in replacement_dict.items():
                if str(key) in str(string):
                    string = string.replace(f'{key}', f' {value} ')
            return string

        def most_common(input_list):
            occurence_count = Counter(input_list)
            return occurence_count.most_common(1)[0][0]

        sentences = DataHandler.lines_from_file(file_path)
        replacement_df = pd.read_csv(offset_path, delimiter="\t", header=None)
        replacement_df.columns = ['source', 'start', 'end', 'term', 'cui']

        replacements = defaultdict(list)
        for i, row in tqdm(replacement_df.iterrows(), total=len(replacement_df)):
            replacements[row['term']].append(row['cui'])

        if "." in replacements:
            del replacements["."]

        replacements = {str(k): str(most_common(vs)) for k, vs in replacements.items()}
        # replaced_sentences = [replace(sentence, replacements) for sentence in tqdm(sentences)]

        replaced_sentences = DataHandler.replace_documents_with_spacy(sentences, replacements)
        DataHandler.save(new_path, "\n".join(replaced_sentences))


DataHandler.julielab_replacements(file_path='E:\AML4DH-DATA\CPG-AMIA2020\Plain Text\cpg-sentences.txt',
                                  offset_path='E:\AML4DH-DATA\offsets\cpg_offsets.tsv',
                                  new_path='E:\AML4DH-DATA\CPG-AMIA2020\Plain Text\cpg-sentences_JULIE.txt')

# DataHandler.julielab_replacements(file_path='E:/AML4DH-DATA/2015_3M_sentences/news_2015_3M-sentences.txt',
#                                   offset_path='E:/AML4DH-DATA/offsets/news_offsets.tsv',
#                                   new_path='E:/AML4DH-DATA/2015_3M_sentences/news_2015_3M-sentences_JULIE.txt')
# DataHandler.tidy_indices('E:/AML4DH-DATA/2015_3M_sentences/deu_news_2015_3M-sentences.txt',
#                          'E:/AML4DH-DATA/2015_3M_sentences/news_2015_3M-sentences.txt')
# DataHandler.split_data('E:/AML4DH-DATA/2015_3M_sentences/news_2015_3M-sentences.txt', 100000,
#                        new_name='E:/AML4DH-DATA/2015_3M_sentences/news_2015_split')
# DataHandler.read_files_and_save_sentences_to_dir("E:\AML4DH-DATA\german_pubmed")