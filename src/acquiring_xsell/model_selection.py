import numpy as np


class MonthSeriesSplit:
    def __init__(self, n_splits=4):
        if n_splits not in [2, 3, 4]:
            raise ValueError("n_splits must be 2, 3, or 4")

        self.n_splits = n_splits
        self.month_col = "month"

    def split(self, data):
        df = data.copy()

        unique_months = np.sort(df[self.month_col].unique())
        n_months = len(unique_months)

        block_size = n_months // self.n_splits

        for i in range(self.n_splits - 1):

            train_months = unique_months[:block_size * (i + 1)]
            valid_months = unique_months[block_size * (i + 1): block_size * (i + 2)]

            train_idx = df[df[self.month_col].isin(train_months)].index
            valid_idx = df[df[self.month_col].isin(valid_months)].index

            yield train_idx, valid_idx