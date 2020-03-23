import numpy as np
import pandas as pd

from load_utils import CheckFeatures as CF, index_to_multiindex


def filter_day_0_mean(df: pd.DataFrame):
    """For a given time, filter and keeps:
        . D-2, D-1, accross all hours: h 1...18
        . D-0, hour 0: h 0

    Alternatives
    ------------
    df.xs('T', level="var", axis=1)
    T.mean(level='D', axis=1)[[-2, -1]]
    T.mean(level=['D', 'h'], axis=1).xs((0, 0), level=['D', 'h'], axis=1)
    """
    columns_to_keep = (df.columns.get_level_values('D') < 0) + (df.columns.get_level_values('h') == 0)
    # Keep only the right columns
    result = df.loc[:, columns_to_keep]
    return result


@CF.validate
def mean_of_var(df: pd.DataFrame, var, test_set=False):
    if test_set:
        df = filter_day_0_mean(df)
    result = df.xs(var, level='var', axis=1)
    result = result.groupby('WF', axis=1).mean()
    result.columns = index_to_multiindex(f'{var}_mean', result.columns)
    return result


@CF.validate
def wind_speed(df: pd.DataFrame, test_set=False):
    if test_set:
        df = filter_day_0_mean(df)
    result = df.xs('U', axis=1, level='var') ** 2 + df.xs('V', axis=1, level='var') ** 2
    result = result.groupby('WF', axis=1).mean()
    result.columns = index_to_multiindex('windspeed', result.columns)
    return result


@CF.validate
def wind_direction(df: pd.DataFrame, test_set=False):
    if test_set:
        df = filter_day_0_mean(df)
    u = df.xs('U', axis=1, level='var')
    v = df.xs('V', axis=1, level='var')
    with np.errstate(invalid="ignore"):
        result = np.arccos(u * v / np.abs(u * v))
    result = result.groupby('WF', axis=1).mean()
    result.columns = index_to_multiindex('angle', result.columns)
    # Deal with potential nan values got in arccos computation.
    result = result.interpolate()
    return result


@CF.validate(result_only=True)
def k_diff(df: pd.DataFrame, k=1, var_level=None):
    """Compute the k-th difference of a given variable. If there is only one level value for the `var` level,
    you can leave that parameter undefined."""
    assert k > 0, f"k should be positive, got {k}"
    if var_level is None:
        level_values = df.columns.get_level_values("var").unique()
        if level_values.size != 1:
            raise ValueError('You did not pass the `var_level` parameter but the df is not explicit; got multiple '
                             'possible level values:' + level_values)
        var_level = level_values[0]
    result = df.xs(var_level, axis=1, level="var").diff(k)
    result.columns = index_to_multiindex(f"{var_level}_{k}_diff", result.columns)
    # Remove NaN in the first line
    result.iloc[0] = result.iloc[1]
    return result


@CF.validate
def all_features(df: pd.DataFrame, get_diff=[1], test_set=False):
    """
    Compute all the features from the input df.
    All other functions implemented in this file should be added to this function!

    Parameters
    ----------
    df:
        The input DataFrame (compliant with CheckFeatures' formatting).
    get_diff:
        Whether to compute the first difference of all the computed features.

    Returns
    -------
    A DataFrame with time as index, and columns having two levels:
        * WF: [1, 2, 3, 4, 5, 6]
        * var: [var_1, ..., var_n]
    """
    # Basic Features
    temperature = mean_of_var(df, "T", test_set=test_set)
    clouds = mean_of_var(df, "CLCT", test_set=test_set)
    speed = wind_speed(df, test_set=test_set)
    direction = wind_direction(df, test_set=test_set)
    features = pd.concat([temperature, clouds, speed, direction], axis=1)

    # Insert your feature function call here!

    # Whether to compute first diff for all features
    variables = features.columns.get_level_values("var").unique()
    for k in get_diff:
        diff = []
        for var in variables:
            var_diff = k_diff(features, var_level=var, k=k)
            diff.append(var_diff)

        features = pd.concat([features] + diff, axis=1)

    return features


def check_isin(value, accepted_values):
    if not value in accepted_values:
        raise ValueError(f"You did not pass a good value. Got: {value} when accepted values are: {accepted_values}.")


if __name__ == '__main__':
    from load_utils import load_data

    df, target = load_data()
    a = wind_direction(df)
    a_diff = k_diff(a)
    features = all_features(df, get_diff=[1, 2])

    # All that for:
    full_df = pd.concat([features, target], axis=1)
    # Puts WF in columns
    full_df = full_df.stack('WF').reset_index(level='WF')
    full_df.columns.set_names('', inplace=True)
