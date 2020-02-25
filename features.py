import pandas as pd
import numpy as np

from load_utils import CheckFeatures as CF


@CF.validate
def mean_of_var(df: pd.DataFrame, var):
    result = df.xs(var, level='var', axis=1).groupby('WF', axis=1).mean()
    result.columns = index_to_multiindex(f'{var}_mean', result.columns)
    return result


@CF.validate
def wind_speed(df: pd.DataFrame):
    result = df.xs('U', axis=1, level='var')**2 + df.xs('V', axis=1, level='var')**2
    result = result.groupby('WF', axis=1).mean()
    result.columns = index_to_multiindex('windspeed', result.columns)
    return result


def check_isin(value, accepted_values):
    if not value in accepted_values:
        raise ValueError(f"You did not pass a good value. Got: {value} when accepted values are: {accepted_values}.")


def index_to_multiindex(new_key, index, level='var'):
    multiindex = pd.MultiIndex.from_arrays([index, [new_key] * len(index)],
                                           names=index.names + [level])
    return multiindex


if __name__ == '__main__':
    from load_utils import load_train_data
    df = load_train_data()
    r = wind_speed(df)
    t = mean_of_var(df, 'T')
