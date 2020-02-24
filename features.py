import pandas as pd
import numpy as np

from load_utils import CheckFeatures
hint = CheckFeatures()


@hint.validate
def mean_of_var(df: pd.DataFrame, var):
    grouped = df.groupby('WF', axis=1)
    #TODO: complete
    grouped.agg()
    pass


if __name__ == '__main__':
    from load_utils import load_train_data
    df = load_train_data()
