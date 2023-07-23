import os.path
from typing import List

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from treevalue import jsonify

from .tokenizer import tokenize


class MarkedTextDataset(Dataset):
    def __init__(self, file: str, text_column: str, data_columns: List[str],
                 tokenizer: str = 'bert', token_size: int = 64):
        _, ext = os.path.splitext(file)
        if ext in {'.csv'}:
            self.df = pd.read_csv(ext)
        elif ext in {'.xls', '.xlsx'}:
            self.df = pd.read_excel(file)
        else:
            raise TypeError(f'Unknown file format - {file!r}.')

        self.texts = self.df[text_column]
        self.tokenizer = tokenizer
        self.token_size = token_size

        self.data_columns = data_columns
        self.column_values = {}
        self.column_labels = {}
        self.column_n_classes = {}
        for dc in data_columns:
            column = self.df[dc]
            uni, values = np.unique(column, return_inverse=True)
            self.column_values[dc] = values
            labels = []
            for i, (tvalue, value) in enumerate(sorted(set(zip(column, values)), key=lambda x: x[1])):
                assert i == value.item()
                labels.append(tvalue)
            self.column_labels[dc] = labels
            self.column_n_classes[dc] = len(uni)

    def __getitem__(self, item):
        text_token = jsonify(tokenize(self.texts[item], self.tokenizer, self.token_size))
        marks = {name: self.column_values[name][item] for name in self.data_columns}
        return text_token, marks

    def __len__(self):
        return len(self.texts)
