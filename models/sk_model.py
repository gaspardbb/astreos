import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from CAPE_CNR_metric import CAPE_CNR_function
from features import all_features
from load_utils import load_data

# Useful module to handle different logging levels and avoid "print" statements.
# You can also specify a minimum verbose with loglevel().
from logzero import logger, loglevel


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


class SkRegressorFull(object):
    """ Apply sklearn regressor to each windfarm separately """

    def __init__(self, sk_model, model_params, nan_handler=handle_nan, train_valid_ratio=0.2, random_state=42):
        self.nan_handler = FunctionTransformer(nan_handler, check_inverse=False)
        self.model = sk_model
        self.model_params = model_params
        self.train_valid_ratio = train_valid_ratio
        self.random_state = random_state
        self.best_regressors = []

    def fit(self, X, Y=None, verbose=0):
        """ Will consider X as the full dataset if Y is None, otherwise behave as sklearn models do """

        handle_nan_trans = FunctionTransformer(handle_nan, check_inverse=False, validate=False)
        steps = [('imputer', handle_nan_trans), ('scaler', StandardScaler()), ('model', self.model())]
        self.pipeline = Pipeline(steps)
        self.scorer = make_scorer(CAPE_CNR_function, greater_is_better=False)
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
            logger.debug(len(full_wfs_df))
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

    def predict(self, X, full=True):
        if not full:
            assert len(self.best_regressors) == 1
            return self.best_regressors[0].predict(X)

        else:
            assert len(X) == len(self.best_regressors)
            predictions = []
            for i, X_wf in enumerate(X):
                predictions.append(self.best_regressors[i].predict(X_wf))
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
    features = all_features(df, get_diff=[1])
    full_df = pd.concat([features, target], axis=1)

    model = SGDRegressor
    parameters = dict(loss='huber', penalty='l2', alpha=0.0001,
                      fit_intercept=True, max_iter=200, tol=0.001,)
    parameters = {f"model__{k}": [v] for k, v in parameters.items()}

    # model = MLPRegressor
    # parameters = {'model__alpha': 10.0 ** np.arange(-5, -4),
    #                'model__hidden_layer_sizes': [(100,) * i for i in range(1, 2)]}

    sk_regressor_full = SkRegressorFull(model, parameters)
    sk_regressor_full.fit(full_df)
