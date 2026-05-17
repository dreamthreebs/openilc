import csv

import numpy as np


class ConfigTable:
    def __init__(self, rows):
        self.rows = [
            {str(key).strip(): self._parse_value(value) for key, value in row.items()}
            for row in rows
        ]
        self.at = self

    @staticmethod
    def _parse_value(value):
        if isinstance(value, (int, float)):
            return value
        value = str(value).strip()
        try:
            as_float = float(value)
        except ValueError:
            return value
        return int(as_float) if as_float.is_integer() else as_float

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row, column = key
            return self.rows[row][column]
        return np.asarray([row[key] for row in self.rows])

    def indices_where(self, column, predicate):
        return [i for i, row in enumerate(self.rows) if predicate(row[column])]


def load_table(value):
    if isinstance(value, ConfigTable):
        return value
    if isinstance(value, (list, tuple)):
        return ConfigTable(value)
    raise TypeError("configuration tables must be CSV-loaded, lists, or ConfigTable")


def load_csv_table(path):
    with open(path) as stream:
        rows = list(csv.DictReader(stream, skipinitialspace=True))
    return ConfigTable(rows)
