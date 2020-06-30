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
    def concat_path_sentences(paths: List[str]) -> List[str]:
        sentences = []
        for path in paths:
            sentences.extend(DataHandler.lines_from_file(path))
        return sentences

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
        DataHandler.save(new_path, "\n".join(replaced_sentences))

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


def replace_stanford_embeddings(emb_path: str, repl_path: str, new_path: str):
    repl_lines = DataHandler.lines_from_file(repl_path)
    look_up = {}
    for line in repl_lines:
        splitted = line.split()
        if len(splitted) == 2:
            look_up[splitted[0]] = splitted[1]
        else:
            print(splitted)

    emb_lines = DataHandler.lines_from_file(emb_path)
    new_lines = []
    for line in emb_lines:
        splitted = line.split()
        if len(splitted) == 301:
            if splitted[0] in look_up:
                repl = look_up[splitted[0]]
            else:
                repl = splitted[0]
            print(repl)
            new_line = f'{repl} {" ".join(splitted[1:])}'
        else:
            new_line = line

        new_lines.append(new_line)

    DataHandler.save(new_path, "\n".join(new_lines))


def reformat_cui2vec(emb_path: str, new_path: str):
    df = pd.read_csv(emb_path)
    df.rename(columns={'Unnamed: 0': 'CUI'}, inplace=True)
    lines = [f'{len(df.index)} {len(df.columns)-1}']
    for i, row in tqdm(df.iterrows(), total=len(df.index)):
        row_list = [str(value) for value in row.values]
        lines.append(f'{row["CUI"]} {" ".join(row_list[1:])}')
    lines.append('')
    print(lines[:2])

    DataHandler.save(new_path, "\n".join(lines))



# DataHandler.julielab_replacements(file_path='E:\AML4DH-DATA\CPG-AMIA2020\Plain Text\cpg-sentences.txt',
#                                   offset_path='E:\AML4DH-DATA\offsets\cpg_offsets.tsv',
#                                   new_path='E:\AML4DH-DATA\CPG-AMIA2020\Plain Text\cpg-sentences_JULIE.txt')


# DataHandler.julielab_replacements(file_path='E:/AML4DH-DATA/JSynCC/jsynncc-sentences.txt',
#                                   offset_path='E:/AML4DH-DATA/offsets/jsynncc_offsets.tsv',
#                                   new_path='E:/AML4DH-DATA/JSynCC/jsynncc-sentences_JULIE.txt')
#
# DataHandler.julielab_replacements(file_path='E:/AML4DH-DATA/german_pubmed/all_sentences.txt',
#                                   offset_path='E:/AML4DH-DATA/offsets/pubmed_offsets.tsv',
#                                   new_path='E:/AML4DH-DATA/german_pubmed/all_sentences_JULIE.txt')


# DataHandler.julielab_replacements(file_path='E:/AML4DH-DATA/2015_3M_sentences/news_2015_3M-sentences.txt',
#                                   offset_path='E:/AML4DH-DATA/offsets/news_offsets.tsv',
#                                   new_path='E:/AML4DH-DATA/2015_3M_sentences/news_2015_3M-sentences_JULIE.txt')
# DataHandler.tidy_indices('E:/AML4DH-DATA/2015_3M_sentences/deu_news_2015_3M-sentences.txt',
#                          'E:/AML4DH-DATA/2015_3M_sentences/news_2015_3M-sentences.txt')
# DataHandler.split_data('E:/AML4DH-DATA/2015_3M_sentences/news_2015_3M-sentences.txt', 100000,
#                        new_name='E:/AML4DH-DATA/2015_3M_sentences/news_2015_split')
# DataHandler.read_files_and_save_sentences_to_dir("E:\AML4DH-DATA\german_pubmed")
# replace_stanford_embeddings('E:/AML4DH-DATA/stanford_cuis_svd_300.txt', 'E:/AML4DH-DATA/NDF/2b_concept_ID_to_CUI.txt', 'E:/AML4DH-DATA/stanford_umls_svd_300.txt')
# reformat_cui2vec('E:/AML4DH-DATA/cui2vec_pretrained.csv', 'E:/AML4DH-DATA/cui2vec_pretrained.txt')