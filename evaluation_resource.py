from abc import ABC, abstractmethod
import os
from typing import Dict, List
from collections import defaultdict


class EvaluationResource(ABC):
    @abstractmethod
    def load_semantics(self, directory: str):
        raise NotImplementedError


class NDFEvaluator(EvaluationResource):
    def __init__(self, from_dir="E:/AML4DH-DATA/NDF"):
        print("initialize UMLSEvaluator...")
        self.may_treat, self.may_prevent, self.reverted_treat, self.reverted_prevent = self.load_semantics(from_dir)

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