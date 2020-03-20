# Load the dataframes, update the columns and indices
import functools
import inspect
from functools import partial

import pandas as pd
import numpy as np


def index_to_multiindex(new_key, index, level='var'):
    multiindex = pd.MultiIndex.from_arrays([index, [new_key] * len(index)],
                                           names=index.names + [level])
    return multiindex


def load_train_data(path_train='data/X_train_v2.csv', time_index_only=True, crop_train_period=True):
    DeprecationWarning("Deprecated. Use `load_data()`.")
    return load_data(path_train=path_train, time_index_only=time_index_only, crop_train_period=crop_train_period)[0]


def load_data(path_train='data/X_train_v2.csv',
              path_test="data/Y_train_sl9m6Jh.csv",
              time_index_only=True, crop_train_period=True):
    """
    Load the training data into a dataframe.

    Parameters
    ----------
    path_train: str
        Path in which to find the X_train csv file.
    time_index_only: bool, optional
        Whether to have 1 or 2 multiindex.
            * If True, the index on axis 0 will be 'Time', and everything else on axis 1.
            * If False, the index on axis 0 will be a multiindex with ['Time', 'WF'] and everything else on axis 1.
            It can be convenient to align with the training data.
    crop_train_period: bool, optional
        Whether to crop the time series to the period: 2018-05-01 -> 2019-01-15.

    Returns
    -------
    df: pd.DataFrame
        A pandas dataframe.
    """
    df = pd.read_csv(path_train, index_col='ID')
    df_target = pd.read_csv(path_test, index_col='ID')

    # Otherwise, the time column is a dumb index column
    df = df.set_index('Time')
    df.index = pd.to_datetime(df.index)

    # Otherwise, dtype=Object=terrible
    df['WF'] = df['WF'].astype('category')
    df['WF'] = df['WF'].cat.rename_categories(range(1, 7))

    # Before transforming the index of the DF, we add the WF information to the target
    df_target.index = df.index
    df_target['WF'] = df['WF']
    # We pivot it so that the WF is in the columns
    df_target = df_target.pivot(columns='WF', values='Production')
    df_target.columns = index_to_multiindex('Production', df_target.columns, level="var")

    # First we define the multiindex we will use
    multiindex = [tuple(col.split('_')) for col in df.columns[1:]]
    for i, (nwp, hour, day, var) in enumerate(multiindex):
        multiindex[i] = (int(nwp[-1]), 0 if day == 'D' else int(day[1:]), int(hour[:2]), var)
    multiindex = pd.MultiIndex.from_tuples(multiindex, names=['NWP', 'D', 'h', 'var'])

    # Then we go through each windfarm
    list_of_groups = []
    list_of_labels = []

    for labels, group in df.groupby('WF'):
        group = group.drop('WF', axis=1)
        group.columns = multiindex
        list_of_labels.append(labels)
        list_of_groups.append(group)

    # And finally we concatenate all WF
    df = pd.concat(list_of_groups, axis=1, keys=list_of_labels, names=['WF'])

    if crop_train_period:
        df = df.loc[pd.to_datetime('2018-05-01 00:00:00'):pd.to_datetime('2019-01-15 23:00:00')]
        df_target = df_target.loc[pd.to_datetime('2018-05-01 00:00:00'):pd.to_datetime('2019-01-15 23:00:00')]

    if not time_index_only:
        df = df.stack(level='WF')

    return df, df_target


def save_multiindex(df: pd.DataFrame, path='save/'):
    """Save the standard multiindex used to check the inputs when doing feature engineering."""
    np.save(f'{path}multiindex.npy', df.columns.to_numpy())


def load_multiindex(path='save/multiindex.npy'):
    """Load a multiindex."""
    return pd.MultiIndex.from_tuples(np.load(path, allow_pickle=True), names=['WF', 'NWP', 'D', 'h', 'var'])


class CheckFeatures:

    path = 'save/multiindex.npy'
    multiindex = load_multiindex(path)

    n_calls = {}
    @staticmethod
    def validate(func=None, *, result_only=False):
        # Handling of default values
        if func is None:
            return partial(CheckFeatures.validate, result_only=result_only)

        # Check if the function has a df parameter, using inspect.signature()
        signature = inspect.signature(func)
        if 'df' not in signature.parameters:
            raise ValueError("A feature function should have a `df` parameter.")

        CheckFeatures.n_calls[func.__name__] = 0

        # Create the decorator
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            this_df = signature.bind(*args, **kwargs).arguments['df']

            if not result_only:

                # Retrieve the df value
                if not isinstance(this_df, pd.DataFrame):
                    raise ValueError(f"You need to supply a DataFrame, but you pass a {type(this_df)}.")

                # Check MultiIndex
                try:
                    if not (this_df.columns == CheckFeatures.multiindex).all():
                        raise ValueError("The df you have passed to this function does not have the right multiindex for "
                                         "the columns.")
                except Exception as e:
                    raise ValueError(f"Your DataFrame do not have the right index. When comparing your column's index "
                                     f"with the reference's index, got exception: {e}")

            # Check the return value
            return_value = func(*args, **kwargs)

            # If it is a DataFrame
            if not isinstance(return_value, pd.DataFrame):
                raise ValueError(f"You need to return a DataFrame, but you return a {type(return_value)}.")

            # If it has the right number of rows
            if not return_value.shape[0] == this_df.shape[0]:
                raise ValueError("You're returning a DataFrame which does not have the same number of rows. "
                                 f"Input: {this_df.shape[0]}. Output: {return_value.shape[0]}.")

            # If it is a MultiIndex, then it should have a level named 'WF'
            if isinstance(return_value.columns, pd.MultiIndex) and not 'WF' in return_value.columns.names:
                raise ValueError("You're returning a DataFrame with a MultiIndex, but it does not have a level named: "
                                 "'WF'.")

            CheckFeatures.n_calls[func.__name__] += 1
            return return_value
        return wrapper




if __name__ == '__main__':
    df_features, df_target = load_data(time_index_only=False)