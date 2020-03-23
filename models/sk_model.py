from typing import List

import numpy as np
import pandas as pd
# Useful module to handle different logging levels and avoid "print" statements.
# You can also specify a minimum verbose with loglevel().
from logzero import logger
from sklearn import clone
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from CAPE_CNR_metric import CAPE_CNR_function
from features import all_features
from load_utils import load_data


def split_data_wf(full_df: pd.DataFrame):
    """ Get separated datasets for each windfarm """
    full_df_wfs = []
    for i in set(full_df.columns.get_level_values('WF')):
        full_df_wfs.append(full_df.xs(i, axis=1, level='WF'))
    return full_df_wfs


def handle_nan(X):
    """ Replace nan using backward fill / forward fill """
    # Looks like the Transformer class of sklearn turns DataFrame into numpy arrays, which do not have the fillna
    # method.
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    Z = X.fillna(method='bfill')
    Z = Z.fillna(method='ffill')
    # if pd.isna(Z).values.any():
    #     In some cases,
    # Z = Z.interpolate()
    return Z


def handle_none_Y(X, Y):
    if Y is None:
        Y = X.xs('Production', level='var', axis=1)
        X = X.drop('Production', level='var', axis=1)

    if Y.isna().values.sum() > 0:
        logger.warning("Some target values are Nan! Removing specific lines...")
        X = X[~Y.isna()]
        Y = Y[~Y.isna()]

    return X, Y


class SkMetaRegressor(object):
    """Base class for regressing on wind farms with Scikit-learn's regressors."""

    def __init__(self, sk_model, model_params, nan_handler=handle_nan, train_valid_ratio=0.2, random_state=42):
        self.nan_handler = FunctionTransformer(nan_handler, check_inverse=False)
        self.model = sk_model
        self.model_params = model_params
        self.train_valid_ratio = train_valid_ratio
        self.random_state = random_state
        self.score_function = CAPE_CNR_function
        self.scorer = make_scorer(self.score_function, greater_is_better=False)

        self.best_regressors = []
        handle_nan_trans = FunctionTransformer(handle_nan, check_inverse=False, validate=False)
        steps = [('imputer', handle_nan_trans), ('scaler', StandardScaler()), ('model', self.model())]
        self.pipeline = Pipeline(steps)

    def fit(self, X, Y=None, *args, **kwargs):
        raise NotImplementedError

    def score(self, X, Y=None, *args, **kwargs):
        raise NotImplementedError

    def predict(self, X, *args, **kwargs):
        raise NotImplementedError


class SkRegressorAll(SkMetaRegressor):
    """Apply sklearn regressor on all windfarms."""

    def __init__(self, sk_model, model_params, nan_handler=handle_nan, train_valid_ratio=0.2, random_state=42):
        super(SkRegressorAll, self).__init__(sk_model, model_params,
                                             nan_handler=nan_handler,
                                             train_valid_ratio=train_valid_ratio,
                                             random_state=random_state)

    def fit(self, X, Y=None, verbose=0, diff_only=True, *args, **kwargs):
        self.grid = GridSearchCV(self.pipeline, param_grid=self.model_params, cv=5, scoring=self.scorer,
                                 verbose=verbose, error_score='raise')

        X, Y = handle_none_Y(X, Y)
        X = X.stack('WF')
        Y = Y.stack('WF')
        if diff_only:
            X = X.filter(like='_diff')

        if hasattr(self, "input_shape"):
            assert self.input_shape == X.shape[1:]
        else:
            pass
        self.diff_only = diff_only  # This value will be set once
        self.input_shape = X.shape[1:]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=self.random_state)

        self.grid.fit(X_train, y_train)
        logger.info("score on test = %3.2f" % (- self.grid.score(X_test, y_test)))

        # self.best_regressors = clone(self.grid.best_estimator_)
        # self.best_regressors['model'].set_params(**extract_params(self.grid.best_params_))

    def predict(self, X, need_unstacking=False, need_filterting=True, *args, **kwargs):
        if need_filterting and self.diff_only:
            X = X.filter(like='_diff')
        if need_unstacking:
            X = X.stack('WF')
        assert X.shape[1:] == self.input_shape
        return self.grid.predict(X)

    def score(self, X, Y=None, *args, **kwargs):
        X, Y = handle_none_Y(X, Y)
        X = X.stack('WF')
        Y = Y.stack('WF')
        predictions = self.predict(X)
        return self.score_function(predictions, Y)


class SkRegressorFull(SkMetaRegressor):
    """ Apply sklearn regressor to each windfarm separately """

    def __init__(self, sk_model, model_params, nan_handler=handle_nan, train_valid_ratio=0.2, random_state=42):
        super(SkRegressorFull, self).__init__(sk_model, model_params, nan_handler=nan_handler,
                                              train_valid_ratio=train_valid_ratio, random_state=random_state)

    def fit(self, X, Y=None, verbose=0):
        """ Will consider X as the full dataset if Y is None, otherwise behave as sklearn models do """

        self.grid = GridSearchCV(self.pipeline, param_grid=self.model_params, cv=5, scoring=self.scorer,
                                 verbose=verbose, error_score='raise')

        self.best_regressors = []

        if Y is not None:
            if Y.isna().sum() > 0:
                logger.warning("Some target values are Nan! Removing specific lines...")
                X = X[~Y.isna()]
                Y = Y[~Y.isna()]
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=self.random_state)
            self.fit_one_(X_train, X_test, y_train, y_test)
            self.best_regressors[-1].fit(X, Y)

        else:
            full_wfs_df = split_data_wf(X)
            self.best_regressors = []
            for i, wf_df in enumerate(full_wfs_df):
                logger.info(f"Fit on WF {i + 1}")
                X_wf = wf_df.drop(['Production'], axis=1)
                Y_wf = wf_df['Production']

                if Y_wf.isna().sum() > 0:
                    logger.warning("Some target values are Nan! Removing specific lines...")
                    X_wf = X_wf[~Y_wf.isna()]
                    Y_wf = Y_wf[~Y_wf.isna()]

                X_train, X_test, y_train, y_test = train_test_split(X_wf, Y_wf, test_size=0.2, random_state=30)

                self.fit_one_(X_train, X_test, y_train, y_test)
                self.best_regressors[-1].fit(X_wf, Y_wf)

    def score(self, X, Y=None) -> List[float]:
        """ Make prediction on X and score it against true value contained either in Y
         or in column(s) 'Production' of X """

        if Y is not None:
            assert len(self.best_regressors) == 1, logger.error(
                "More than one regressor, cannot decide which one to use")
            X_wfs, Y_wfs = X, Y

        else:
            full_df_wfs = split_data_wf(X)
            assert len(full_df_wfs) == len(self.best_regressors), logger.error(
                f"Different number of regressors ({len(self.best_regressors)}) and datasets ({len(full_df_wfs)})")
            assert 'Production' in full_df_wfs[0], logger.error("No column 'Prediction in X, cannot score prediction")
            X_wfs = list(map(lambda df: df.drop('Production', axis=1), full_df_wfs))
            Y_wfs = list(map(lambda df: df['Production'], full_df_wfs))

        predictions = self.predict(X_wfs)
        scores = []
        for prediction, Y_wf in zip(predictions, Y_wfs):
            scores.append(self.score_function(prediction, Y_wf))

        return scores

    def predict(self, X, full=True) -> List[np.array]:
        if not full:
            assert len(self.best_regressors) == 1
            return [np.clip(self.best_regressors[0].predict(X), a_min=0, a_max=np.inf)]

        else:
            if type(X) == pd.DataFrame:
                X = split_data_wf(X)
                assert 'Production' not in X[0], logger.error(
                    "Column 'Prediction' should not appear")
            assert len(X) == len(self.best_regressors), logger.error(
                    f"Different number of regressors ({len(self.best_regressors)}) and datasets ({len(X)})")
            predictions = []
            for i, X_wf in enumerate(X):
                predictions.append(np.clip(self.best_regressors[i].predict(X_wf), a_min=0, a_max=np.inf))
            return predictions

    def fit_one_(self, X_train, X_test, y_train, y_test):
        try:
            # assert not pd.isna(X_train).values.any(), f"Got NaN values in X_train."
            # assert not pd.isna(y_train).values.any(), f"Got NaN values in y_train."
            self.grid.fit(X_train, y_train)
        except ValueError as e:
            print(
                "Make sure you named your model parameters with names starting with 'model__, error may come from that'")
            raise e
        logger.info("score on test = %3.2f" % (- self.grid.score(X_test, y_test)))
        logger.debug(self.grid.best_params_)
        self.best_regressors.append(clone(self.pipeline))
        self.best_regressors[-1]['model'].set_params(**extract_params(self.grid.best_params_))


def extract_params(best_params):
    params = {}
    for k, v in best_params.items():
        if '__' in k:
            k = k[k.find('__') + 2:]
        params[k] = v
    return params


if __name__ == '__main__':
    print("Load dataset...")
    df, target = load_data()
    features = all_features(df, get_diff=[1], test_set=True)
    full_df = pd.concat([features, target], axis=1)

    model = SGDRegressor
    parameters = dict(loss='huber', penalty='l2', alpha=0.0001,
                      fit_intercept=True, max_iter=200, tol=0.001, )
    parameters = {f"model__{k}": [v] for k, v in parameters.items()}

    # model = MLPRegressor
    # parameters = {'model__alpha': 10.0 ** np.arange(-5, -4),
    #                'model__hidden_layer_sizes': [(100,) * i for i in range(1, 2)]}

    sk_regressor_full = SkRegressorFull(model, parameters)
    sk_regressor_full.fit(full_df)

    logger.info("Test prediction on full dataset")
    full_df_wfs = list(map(lambda df: df.drop('Production', axis=1), split_data_wf(full_df)))
    sk_regressor_full.predict(full_df_wfs)

    logger.info("Test scoring on full dataset")
    print(sk_regressor_full.score(full_df))

    sk_regressor_all = SkRegressorAll(model, parameters)
    sk_regressor_all.fit(full_df, diff_only=True)
    logger.info("Test scoring on full dataset -- ALL model")
    print(sk_regressor_all.score(full_df))
