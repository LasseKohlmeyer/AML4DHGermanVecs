import os
import tqdm


class NDFEvaluator:
    def __init__(self, from_dir="E:/AML4DH-DATA/NDF"):
        print("initialize UMLSEvaluator...")
        self.may_treat, self.may_prevent = self.load_ndf_semantics(directory=from_dir)

    def load_ndf_semantics(self, directory):
        def load_file(file_name: str):
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

        may_treat_dict = load_file(file_name="may_treat_cui.txt")
        may_prevent_dict = load_file(file_name="may_prevent_cui.txt")

        return may_treat_dict, may_prevent_dict

eval = NDFEvaluator()