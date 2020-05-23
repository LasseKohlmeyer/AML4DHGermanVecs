import json
from abc import ABC, abstractmethod
import os
from typing import Dict, List
from collections import defaultdict
import pandas as pd
from sklearn import preprocessing

class EvaluationResource(ABC):
    @abstractmethod
    def load_semantics(self, directory: str):
        raise NotImplementedError

    def save_as_json(self, path: str):
        data = self.__dict__
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=0)

    @abstractmethod
    def load_from_json(self, path: str):
        raise NotImplementedError


class NDFEvaluator(EvaluationResource):
    def __init__(self, json_path: str = None, from_dir: str = None):
        print(f"initialize {self.__class__.__name__}...")
        if from_dir:
            print(f"initialize {self.__class__.__name__}... Load dir")
            self.may_treat, self.may_prevent, self.reverted_treat, self.reverted_prevent = self.load_semantics(from_dir)
        if json_path:
            print(f"initialize {self.__class__.__name__}... Load json")
            self.may_treat, self.may_prevent, self.reverted_treat, self.reverted_prevent = self.load_from_json(json_path)

    def load_semantics(self, directory: str):
        def load_file(file_name: str) -> Dict[str, List[str]]:
            with open(os.path.join(directory, file_name), encoding="utf-8") as f:
                data = f.read()
            dictionary = {}
            for line in data.split('\n'):
                line_parsed = line.split(':')
                # print(line_parsed)
                if len(line_parsed) == 2:
                    key, values = line_parsed[0], line_parsed[1]
                    dictionary[key] = [c for c in values.split(",") if c]
            return dictionary

        # todo: merge with other revert
        def revert(input_dict: Dict[str, List[str]]) -> Dict[str, List[str]]:
            reverted_dict = defaultdict(list)
            for key, values in input_dict.items():
                for value in values:
                    reverted_dict[value].append(key)
            return reverted_dict

        may_treat_dict = load_file(file_name="may_treat_cui.txt")
        may_prevent_dict = load_file(file_name="may_prevent_cui.txt")

        may_treat_reverted_dict = revert(may_treat_dict)
        may_prevent_reverted_dict = revert(may_treat_dict)
        return may_treat_dict, may_prevent_dict, may_treat_reverted_dict, may_prevent_reverted_dict

    def load_from_json(self, path: str):
        with open(path, 'r', encoding='utf-8') as file:
            data = json.loads(file.read())
        return data["may_treat"], data["may_prevent"], data["reverted_treat"], data["reverted_prevent"]


class SRSEvaluator(EvaluationResource):
    def __init__(self, json_path: str = None, from_dir: str = None):
        if from_dir:
            print(f"initialize {self.__class__.__name__}... Load dir")
            self.human_relatedness, self.human_similarity_cont, self.human_relatedness_cont = self.load_semantics(from_dir)
        if json_path:
            print(f"initialize {self.__class__.__name__}... Load json")
            self.human_relatedness, self.human_similarity_cont, self.human_relatedness_cont = self.load_from_json(json_path)

    def load_semantics(self, directory: str):
        def load_file(file_name: str) -> pd.DataFrame:
            path = os.path.join(directory, file_name)
            df = pd.read_csv(path)
            return df

        def parse_file(file_name: str):
            df = load_file(file_name)
            df[["Mean"]] = (df[["Mean"]] - df[["Mean"]].min()) / (
                        df[["Mean"]].max() - df[["Mean"]].min())
            nested_dictionary = defaultdict(lambda: defaultdict(dict))
            for i, row in df.iterrows():
                nested_dictionary[row["CUI1"]][row["CUI2"]] = row["Mean"]
                nested_dictionary[row["CUI2"]][row["CUI1"]] = row["Mean"]

            return nested_dictionary

        human_relatedness_dict = parse_file(file_name="MayoSRS.csv")
        human_similarity_cont_dict = parse_file(file_name="UMNSRS_similarity.csv")
        human_relatedness_cont_dict = parse_file(file_name="UMNSRS_relatedness.csv")

        return human_relatedness_dict, human_similarity_cont_dict, human_relatedness_cont_dict

    def load_from_json(self, path: str):
        with open(path, 'r', encoding='utf-8') as file:
            data = json.loads(file.read())
        return data["human_relatedness"], data["human_similarity_cont"], data["human_relatedness_cont"]


def main():
    pass

if __name__ == "__main__":
    main()