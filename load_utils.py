# Load the dataframes, update the columns and indices
import functools
import inspect
import os
from functools import partial

import numpy as np
import pandas as pd
from logzero import logger

from utils.utils import get_project_root


def index_to_multiindex(new_key, index, level='var'):
    multiindex = pd.MultiIndex.from_arrays([index, [new_key] * len(index)],
                                           names=index.names + [level])
    return multiindex


def load_index(path_X='data/X_train_v2.csv'):
    """Returns a DataFrame with one column: 'ID' and index a MultiIndex ('Time', 'WF')."""
    path_X = os.path.join(get_project_root(), path_X)
    df_X = pd.read_csv(path_X)
    df_X = df_X[['ID', 'Time', 'WF']]
    df_X['Time'] = pd.to_datetime(df_X['Time'])

    df_X['WF'] = df_X['WF'].astype('category')
    df_X['WF'] = df_X['WF'].cat.rename_categories(range(1, 7))

    df_X = df_X.set_index(['Time', 'WF'])
    df_X['ID'] = df_X['ID'].astype('int')
    return df_X


def add_ID_column(predictions, path_X='data/X_train_v2.csv'):
    """Add an "ID" column, where such column was fetched from a CSV. Concatenation occurs according to time and WF
    id."""
    df_id = load_index(path_X)
    if not isinstance(predictions, pd.Series):
        predictions = predictions['Production']
    difference = predictions.index.difference(df_id.index)
    if len(difference) > 0:
        logger.warning(f'These values are in the prediction Series but not in the ID Serie ({len(difference)})\n'
                    f'{difference}')
        logger.warning('DROPPING THEM.')
    result = pd.concat([df_id['ID'], predictions], axis=1, keys=['ID', 'Production'])
    result = result[~result['ID'].isna()]
    result['ID'] = result['ID'].astype('int')
    return result


def load_train_data(path_train='data/X_train_v2.csv', time_index_only=True, crop_train_period=True):
    DeprecationWarning("Deprecated. Use `load_data()`.")
    return load_data(path_X=path_train, time_index_only=time_index_only, crop_train_period=crop_train_period)[0]


def load_data(path_X='data/X_train_v2.csv',
              path_Y="data/Y_train_sl9m6Jh.csv",
              time_index_only=True, crop_train_period=True):
    """
    Load the training data into a dataframe.

    Parameters
    ----------
    path_X: str
        Path in which to find the X_train csv file.
    path_Y: str, optional
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
    path_X = os.path.join(get_project_root(), path_X)
    df_X = pd.read_csv(path_X, index_col='ID')
    df_X.index = df_X.index.astype('int')

    # Otherwise, dtype=Object=terrible
    df_X['WF'] = df_X['WF'].astype('category')
    df_X['WF'] = df_X['WF'].cat.rename_categories(range(1, 7))

    # Oterwise, dtype=str
    df_X['Time'] = pd.to_datetime(df_X['Time'])

    # Before transforming the index of the DF, we add the WF information to the target if any
    if path_Y is not None:
        path_Y = os.path.join(get_project_root(), path_Y)
        df_Y = pd.read_csv(path_Y, index_col='ID')
        df_Y.index = df_Y.index.astype('int')
        # To be sure X and Y share the same ID, the join is done on the ID index.
        full_df = pd.concat([df_X, df_Y], axis=1)

        df_Y = full_df[['Time', 'WF', 'Production']]
        # We pivot it so that the WF is in the columns
        # df_Y = df_Y.pivot(columns='WF', values='Production')
        df_Y = df_Y.set_index(['Time', 'WF'])
        df_Y.columns = df_Y.columns.set_names('var')
        df_Y = df_Y.unstack('WF')
        df_Y.columns = df_Y.columns.reorder_levels(['WF', 'var'])

        df_X = full_df.drop('Production', axis=1)

    # Otherwise, the time column is a dumb index column
    df_X = df_X.set_index('Time')

    # First we define the multiindex we will use
    multiindex = [tuple(col.split('_')) for col in df_X.columns[1:]]
    for i, (nwp, hour, day, var) in enumerate(multiindex):
        multiindex[i] = (int(nwp[-1]), 0 if day == 'D' else int(day[1:]), int(hour[:2]), var)
    multiindex = pd.MultiIndex.from_tuples(multiindex, names=['NWP', 'D', 'h', 'var'])

    # Then we go through each windfarm
    list_of_groups = []
    list_of_labels = []

    for labels, group in df_X.groupby('WF'):
        group = group.drop('WF', axis=1)
        group.columns = multiindex
        list_of_labels.append(labels)
        list_of_groups.append(group)

    # And finally we concatenate all WF
    df_X = pd.concat(list_of_groups, axis=1, keys=list_of_labels, names=['WF'])

    if crop_train_period:
        df_X = df_X.loc[pd.to_datetime('2018-05-01 00:00:00'):pd.to_datetime('2019-01-15 23:00:00')]
        if path_Y is not None:
            df_Y = df_Y.loc[pd.to_datetime('2018-05-01 00:00:00'):pd.to_datetime('2019-01-15 23:00:00')]

    if not time_index_only:
        df_X = df_X.stack(level='WF')

    if path_Y is not None:
        return df_X, df_Y
    return df_X


def save_multiindex(df: pd.DataFrame, path='save/'):
    """Save the standard multiindex used to check the inputs when doing feature engineering."""
    np.save(f'{path}multiindex.npy', df.columns.to_numpy())


def load_multiindex(path='save/multiindex.npy'):
    """Load a multiindex."""
    return pd.MultiIndex.from_tuples(np.load(path, allow_pickle=True), names=['WF', 'NWP', 'D', 'h', 'var'])


class CheckFeatures:
    path = os.path.join(get_project_root(), 'save/multiindex.npy')
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
                        raise ValueError(
                            "The df you have passed to this function does not have the right multiindex for "
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
    df_features = load_data(path_X='data/X_test_v2.csv', path_Y=None)
