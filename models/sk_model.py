from typing import List, Dict

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
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

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


def _wf_to_index(df: pd.DataFrame):
    """Puts the WF level to the index and raise an error if there is none.
    If it is a Serie, returns the DF."""
    if isinstance(df, pd.Series):
        return df
    if 'WF' in df.columns.names:
        df = df.stack('WF', dropna=False)
    elif 'WF' not in df.index.names:
        raise ValueError('Could not find a `WF` level, neither in the index nor in the columns.')
    return df


def handle_none_Y(X, Y):
    """This handle a Y set to None but X having a 'Production' var.
    Returns df having WF in the index."""

    # Some sklearn functions turn Y into an array.
    if isinstance(Y, np.ndarray):
        Y = pd.Series(Y, index=X.index)

    # MultiIndex does not fit so well with functions
    if Y is None:
        if isinstance(X.columns, pd.MultiIndex):
            Y = X.xs('Production', level='var', axis=1)
        else:
            Y = X.loc[:, 'Production']

    if 'Production' in X.columns.get_level_values(level='var'):
        if isinstance(X.columns, pd.MultiIndex):
            X = X.drop('Production', level='var', axis=1)
        else:
            X = X.drop('Production', axis=1)

    X = _wf_to_index(X)
    Y = _wf_to_index(Y)

    assert X.shape[0] == Y.shape[0]

    if Y.isna().values.sum() > 0:
        logger.warning("Some target values are Nan! Removing specific lines...")
        X = X[~Y.isna()]
        Y = Y[~Y.isna()]

    assert X.shape[0] == Y.shape[0]
    return X, Y


class SkRegressorMeta(object):
    """Base class for regressing on wind farms with Scikit-learn's regressors."""
    _estimator_type = "regressor"

    def __init__(self, sk_model, model_params, nan_handler=handle_nan, train_valid_ratio=0.2, random_state=42):
        self.nan_handler_untransformed = handle_nan
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

    def get_params(self, deep=True):
        return dict(sk_model=self.model,
                    model_params=self.model_params,
                    nan_handler=self.nan_handler_untransformed,
                    train_valid_ratio=self.train_valid_ratio,
                    random_state=self.random_state)

    def fit(self, X, Y=None, *args, **kwargs):
        raise NotImplementedError

    def score(self, X, Y=None, individual=False) -> Dict[int, float] or float:
        """ Make prediction on X and score it against true value contained either in Y
         or in column(s) 'Production' of X """
        X, Y = handle_none_Y(X, Y)
        predictions = self.predict(X)
        if individual:
            wf_ids = X.index.get_level_values('WF').unique()
            scores = {wf_id: self.score_function(predictions.xs(wf_id, level='WF'), Y.xs(wf_id, level='WF')) for wf_id
                      in wf_ids}
        else:
            scores = self.score_function(Y, predictions)
        return scores

    def predict(self, X, *args, **kwargs) -> pd.Series:
        raise NotImplementedError


class SkRegressorAll(SkRegressorMeta):
    """Apply sklearn regressor on all windfarms."""

    def __init__(self, sk_model, model_params, nan_handler=handle_nan, train_valid_ratio=0.2, random_state=42):
        super(SkRegressorAll, self).__init__(sk_model, model_params,
                                             nan_handler=nan_handler,
                                             train_valid_ratio=train_valid_ratio,
                                             random_state=random_state)

    def fit(self, X, Y=None, verbose=0, diff_only=True, *args, **kwargs):
        if 'Production' in X.columns.get_level_values('var'):
            Y = None

        self.grid = GridSearchCV(self.pipeline, param_grid=self.model_params, cv=5, scoring=self.scorer,
                                 verbose=verbose, error_score='raise')

        X, Y = handle_none_Y(X, Y)

        if diff_only:
            X = X.filter(like='_diff')

        if hasattr(self, "input_shape"):
            assert self.input_shape == X.shape[1:]
        else:
            self.diff_only = diff_only  # This value will be set once
            self.input_shape = X.shape[1:]

        assert X.shape[0] == Y.shape[0]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=self.random_state)

        self.grid.fit(X_train, y_train)
        logger.info("score on test = %3.2f" % (- self.grid.score(X_test, y_test)))

        # self.best_regressors = clone(self.grid.best_estimator_)
        # self.best_regressors['model'].set_params(**extract_params(self.grid.best_params_))

    def predict(self, X, *args, **kwargs):
        if self.diff_only:
            X = X.filter(like='_diff')
        if 'WF' in X.columns.names:
            X = X.stack('WF')
        assert X.shape[1:] == self.input_shape
        prediction = self.grid.predict(X)
        prediction = pd.Series(prediction, index=X.index)
        return prediction


class SkRegressorIndividual(SkRegressorMeta):
    """ Apply sklearn regressor to each windfarm separately """

    def __init__(self, sk_model, model_params, nan_handler=handle_nan, train_valid_ratio=0.2, random_state=42):
        super(SkRegressorIndividual, self).__init__(sk_model, model_params, nan_handler=nan_handler,
                                                    train_valid_ratio=train_valid_ratio, random_state=random_state)

    def fit(self, X, Y=None, verbose=0):
        """ Will consider X as the full dataset if Y is None, otherwise behave as sklearn models do """
        if 'Production' in X.columns.get_level_values('var'):
            Y = None

        X: pd.DataFrame
        Y: pd.DataFrame

        self.grid = GridSearchCV(self.pipeline, param_grid=self.model_params, cv=5, scoring=self.scorer,
                                 verbose=verbose, error_score='raise')

        X, Y = handle_none_Y(X, Y)
        # At this point, both X and Y have a `WF` entry in the index.
        wf_ids = X.index.get_level_values('WF').unique()

        self.best_regressors = {}
        for wf_id in wf_ids:
            logger.info(f"Fit on WF {wf_id}")
            X_wf = X.xs(wf_id, level='WF')
            Y_wf = Y.xs(wf_id, level='WF')

            assert Y_wf.isna().sum() == 0, "This should have been handled before..."

            X_train, X_test, y_train, y_test = train_test_split(X_wf, Y_wf, test_size=0.2, random_state=30)

            self.fit_one_(X_train, X_test, y_train, y_test, wf_id)
            self.best_regressors[wf_id].fit(X_wf, Y_wf)

    def fit_one_(self, X_train, X_test, y_train, y_test, wf_id):
        try:
            self.grid.fit(X_train, y_train)
        except ValueError as e:
            print(
                "Make sure you named your model parameters with names starting with 'model__, error may come from that'")
            raise e
        logger.info("score on test = %3.2f" % (- self.grid.score(X_test, y_test)))
        logger.debug(self.grid.best_params_)
        self.best_regressors[wf_id] = clone(self.pipeline)
        self.best_regressors[wf_id]['model'].set_params(**extract_params(self.grid.best_params_))


    def predict(self, X) -> pd.Series:
        X = _wf_to_index(X)
        wf_ids = X.index.get_level_values('WF').unique()
        predictions = {}

        for wf_id in wf_ids:
            X_wf = X.xs(wf_id, level='WF')
            if 'Production' in X_wf.columns:
                X_wf = X_wf.drop('Production', axis=1)
            Y_wf = np.clip(self.best_regressors[wf_id].predict(X_wf), a_min=0, a_max=np.inf)
            predictions[wf_id] = pd.Series(Y_wf, index=X_wf.index)

        predictions = pd.DataFrame.from_dict(predictions)
        predictions.columns = predictions.columns.set_names('WF')
        predictions = predictions.stack('WF')
        return predictions


def extract_params(best_params):
    params = {}
    for k, v in best_params.items():
        if '__' in k:
            k = k[k.find('__') + 2:]
        params[k] = v
    return params


def stack_predict(model: StackingRegressor, X):
    """
    StackingRegressor does not handle Series output by predict method of bases estimators. Had to recode that.
    """
    from sklearn.utils.validation import check_is_fitted
    check_is_fitted(model)

    predictions = [
            getattr(est, meth)(X)
            for est, meth in zip(model.estimators_, model.stack_method_)
            if est != 'drop'
        ]
    predictions = _custom_concatenate_predictions(model, X, predictions)

    predictions = model.final_estimator_.predict(predictions)
    return predictions


def _custom_concatenate_predictions(model, X, predictions):
    X_meta = []
    for est_idx, preds in enumerate(predictions):
        # case where the the estimator returned a 1D array
        if preds.ndim == 1:
            if isinstance(preds, pd.Series):
                X_meta.append(preds.values.reshape(-1, 1))
            else:
                X_meta.append(preds.reshape(-1, 1))
        else:
            if (model.stack_method_[est_idx] == 'predict_proba' and
                    len(model.classes_) == 2):
                X_meta.append(preds[:, 1:])
            else:
                X_meta.append(preds)
    if model.passthrough:
        X_meta.append(X)

    return np.hstack(X_meta)


if __name__ == '__main__':
    print("Load dataset...")
    df, target = load_data()
    features = all_features(df, get_diff=[1], test_set=True)
    full_df = pd.concat([features, target], axis=1)

    # model = SGDRegressor
    # parameters = dict(loss='huber', penalty='l2', alpha=0.0001,
    #                   fit_intercept=True, max_iter=200, tol=0.001, )
    # parameters = {f"model__{k}": [v] for k, v in parameters.items()}

    model = MLPRegressor
    parameters = {'model__alpha': 10.0 ** np.arange(-5, -4),
                   'model__hidden_layer_sizes': [(100,) * i for i in range(1, 2)]}

    logger.info("Fit Individual Regressors")
    sk_regressor_individual = SkRegressorIndividual(model, parameters)
    sk_regressor_individual.fit(full_df)
    logger.info("Scoring for Individual")
    print(sk_regressor_individual.score(full_df, individual=False))

    logger.info("Fit All Regressor")
    sk_regressor_all = SkRegressorAll(model, parameters)
    sk_regressor_all.fit(full_df, diff_only=True)
    logger.info("Scoring for All")
    print(sk_regressor_all.score(full_df))

    model = StackingRegressor([('All', SkRegressorAll(model, parameters)),
                               ('Individual', SkRegressorIndividual(model, parameters))],
                              LinearRegression(normalize=True),
                              cv=5, passthrough=True)

    X, Y = handle_none_Y(full_df, None)
    X = handle_nan(X)
    model.fit(X, Y)
    # This does not work
    # X_pred = model.predict(X)
    y_pred_combined = stack_predict(model, X)
    y_pred_combined = pd.Series(y_pred_combined, index=X.index)
    print(CAPE_CNR_function(Y, y_pred_combined))

    y_pred_individual = sk_regressor_individual.predict(X)
    all_y = pd.concat([Y, y_pred_combined, y_pred_individual], axis=1,
                      keys=['True', 'Pred_combined', 'Pred_individual'])
    all_y = all_y.reset_index()

    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.style.use("ggplot")

    g = sns.FacetGrid(all_y, row='WF')
    g = g.map(plt.plot, "Time", 'True', label='True', color="r")
    g = g.map(plt.plot, "Time", 'Pred_combined', label='Pred, Combined', color="b")
    g = g.map(plt.plot, "Time", 'Pred_individual', label='Pred, Individual', color="g")

    plt.legend()
    plt.show()


