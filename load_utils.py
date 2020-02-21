# Load the dataframes, update the columns and indices

import pandas as pd


def load_train_data(path='data/X_train_v2.csv', time_index_only=True):
    """
    Load the training data into a dataframe.

    Parameters
    ----------
    path: str
        Path in which to find the X_train csv file.
    time_index_only: bool, optional
        Whether to have 1 or 2 multiindex.
            * If True, the index on axis 0 will be 'Time', and everything else on axis 1.
            * If False, the index on axis 0 will be a multiindex with ['Time', 'WF'] and everything else on axis 1.

    Returns
    -------
    df: pd.DataFrame
        A pandas dataframe.
    """
    df = pd.read_csv(path, index_col='ID')

    # Otherwise, the time column is a dumb index column
    df = df.set_index('Time')
    df.index = pd.to_datetime(df.index)

    # Otherwise, dtype=Object=terrible
    df['WF'] = df['WF'].astype('category')
    df['WF'] = df['WF'].cat.rename_categories(range(1, 7))

    if not time_index_only:
        # Special indexing having time AND WF id
        index_wf_time = pd.MultiIndex.from_frame(df.reset_index()[['Time', 'WF']])
        df = df.set_index(index_wf_time)
        df = df.drop('WF', axis=1)

    # First we define the multiindex we will use
    columns = df.columns[1:] if time_index_only else df.columns
    multiindex = [tuple(col.split('_')) for col in columns]
    for i, (nwp, hour, day, var) in enumerate(multiindex):
        multiindex[i] = (int(nwp[-1]), 0 if day == 'D' else int(day[1:]), int(hour[:2]), var)
    multiindex = pd.MultiIndex.from_tuples(multiindex, names=['NWP', 'D', 'h', 'var'])

    if time_index_only:
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

    else:
        df.columns = multiindex

    return df