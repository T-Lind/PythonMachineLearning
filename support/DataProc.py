import numpy as np
from pandas import DataFrame
from sklearn.model_selection import StratifiedShuffleSplit


def split_train_test(data: DataFrame, catName: str, test_ratio=0.2, random_state=42):
    global strat_train_set, strat_test_set

    split = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_state)

    for train_index, test_index in split.split(data, data[catName]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]

    return strat_train_set, strat_test_set

