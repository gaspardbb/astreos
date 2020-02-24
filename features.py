import pandas as pd
import numpy as np

from load_utils import CheckFeatures

@CheckFeatures.validate
def mean_of_var(df: pd.DataFrame, var):
    grouped = df.groupby('WF', axis=1)
    #TODO: complete
    return df.iloc[:, [1,2]]


if __name__ == '__main__':
    from load_utils import load_train_data
    df = load_train_data()
