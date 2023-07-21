from abc import ABC, abstractmethod
from multiprocessing import Pool
from typing import List, Dict

import pandas as pd

class AbstractContentTokenizer(ABC):
    def __init__(self, content_column, n_cores=1):
        self.max_len = 1
        self.content_map = self._create_initial_dict(content_column)
        self.n_cores = n_cores

    @abstractmethod
    def _create_initial_dict(self, content: pd.Series) -> Dict[str, int]:
        pass

    @abstractmethod
    def entry_to_id_list(self, content_string: str) -> List[int]:
        pass

    def convert_column_to_ids(self, column: pd.Series) -> pd.Series:
        with Pool(self.n_cores) as p:
            split = pd.np.array_split(column, self.n_cores)
            res = pd.concat(p.map(self.entry_to_id_list, split))
        return res

    @property
    def max_seq_len(self):
        return self.max_len

    @property
    def num_classes(self):
        return len(self.content_map)


class FullStringToOneIdContentTokenizer(AbstractContentTokenizer):
    def _create_initial_dict(self, content):
        return {category: index for index, category in enumerate(set(content), 1)}

    def entry_to_id_list(self, content_string):
        return [self.content_map[content_string]]


class MultiHotContentTokenizer(AbstractContentTokenizer):
    def __init__(self, content_column, split_symbol="|", n_cores=8):
        self.split_symbol = split_symbol
        super().__init__(content_column, n_cores)

    def split(self, string):
        return string.split(self.split_symbol)

    def _create_initial_dict(self, content):
        content = content.unique()
        self.max_len = max(len(self.split(co)) for co in content)
        unique_categories = {category for content_entry in content for category in self.split(content_entry)}
        return {category: index for index, category in enumerate(sorted(unique_categories), 1)}

    def entry_to_id_list(self, content_string):
        return [self.content_map[category] for category in self.split(content_string)]
