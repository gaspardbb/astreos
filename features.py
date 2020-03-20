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


@CF.validate
def wind_direction(df: pd.DataFrame):
    u = df.xs('U', axis=1, level='var')
    v = df.xs('V', axis=1, level='var')
    result = np.arccos(u*v/np.abs(u*v))
    result = result.groupby('WF', axis=1).mean()
    result.columns = index_to_multiindex('angle', result.columns)
    return result


@CF.validate(result_only=True)
def first_diff(df: pd.DataFrame, var_level=None):
    """Compute the first difference of a given variable. If there is only one level value for the `var` level,
    you can leave that parameter undefined."""
    if var_level is None:
        level_values = df.columns.get_level_values("var").unique()
        if level_values.size != 1:
            raise ValueError('You did not pass the `var_level` parameter but the df is not explicit; got multiple '
                             'possible level values:' + level_values)
        var_level = level_values[0]
    result = df.xs(var_level, axis=1, level="var").diff()
    result.columns = index_to_multiindex(f"{var_level}_diff", result.columns)
    return result


@CF.validate
def all_features(df: pd.DataFrame, get_first_diff=True):
    """
    Compute all the features from the input df.
    All other functions implemented in this file should be added to this function!

    Parameters
    ----------
    df:
        The input DataFrame (compliant with CheckFeatures' formatting).
    get_first_diff:
        Whether to compute the first difference of all the computed features.

    Returns
    -------
    A DataFrame with time as index, and columns having two levels:
        * WF: [1, 2, 3, 4, 5, 6]
        * var: [var_1, ..., var_n]
    """
    # Basic Features
    temperature = mean_of_var(df, "T")
    clouds = mean_of_var(df, "CLCT")
    speed = wind_speed(df)
    direction = wind_direction(df)
    features = pd.concat([temperature, clouds, speed, direction], axis=1)

    # Insert your feature function call here!

    # Whether to compute first diff for all features
    if get_first_diff:
        diff = []
        variables = features.columns.get_level_values("var").unique()
        for var in variables:
            var_diff = first_diff(features, var_level=var)
            diff.append(var_diff)

        features = pd.concat([features] + diff, axis=1)

    return features



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
    a = wind_direction(df)
    a_diff = first_diff(a)
    features = all_features(df)
